"""
CLI Application for Indexing and Querying Mechanical Keyboard Switch Scores

This application uses LlamaIndex to create a vector index of PDF documents containing
subjective scores and reviews of mechanical keyboard switches. It integrates with:
- Ollama for local embeddings (bge-m3) and inference (deepseek-r1)
- ChromaDB for persistent vector storage via HTTP client
- Click for command-line interface

The PDFs are sourced from the switch-scores repository and contain sections on
push feel, wobble, sound, context, other factors, and score tables.
"""

import logging
from typing import List, cast

import chromadb
import click
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, PromptTemplate
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure logging to INFO level for operational visibility
# This allows tracking of ingestion progress, errors, and query results
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global LlamaIndex settings for embeddings and LLM
# Using Ollama for local, private processing without external API calls
# bge-m3 is a multilingual embedding model suitable for text retrieval
# deepseek-r1 is a reasoning model for generating natural language responses
Settings.embed_model = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")
Settings.llm = Ollama(model="deepseek-r1:latest", request_timeout=120.0, context_window=128000)


@click.group()
def cli() -> None:
    """CLI for indexing and querying switch scores with LlamaIndex."""
    pass


@cli.command()
@click.option("--data-dir", default="datalake/switch-scores", help="Directory containing PDF files.")
@click.option("--collection-name", default="switch_scores", help="ChromaDB collection name.")
@click.option("--batch-size", default=10, help="Number of PDFs to process per batch.")
def ingest(data_dir: str, collection_name: str, batch_size: int) -> None:
    """
    Ingest PDF files into ChromaDB vector index in batches.

    This command:
    1. Connects to ChromaDB server running on localhost:8000
    2. Creates or retrieves a collection for storing vectors
    3. Loads PDF documents in batches, extracting text content
    4. Embeds the text using bge-m3 via Ollama and adds to index incrementally
    5. Stores embeddings in ChromaDB for later retrieval

    Failed PDFs are logged as warnings and skipped. Batching helps with memory management.
    """
    # Initialize ChromaDB HTTP client for server-mode persistence
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    try:
        chroma_client.heartbeat()  # Verify ChromaDB server is accessible
    except Exception as e:
        logger.error(f"ChromaDB not running: {e}")
        return

    # Get or create collection; ChromaDB handles persistence across sessions
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Use SimpleDirectoryReader to scan directory for PDFs
    # required_exts ensures only PDF files are processed
    reader = SimpleDirectoryReader(data_dir, required_exts=[".pdf"])
    total_files = len(reader.input_files)
    logger.info(f"Found {total_files} PDF files to process in batches of {batch_size}.")

    for i in range(0, total_files, batch_size):
        batch_files = reader.input_files[i : i + batch_size]
        documents: List = []
        for file_path in batch_files:
            try:
                # Load individual PDF to handle parsing errors per file
                docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                documents.extend(docs)
                logger.info(f"Loaded {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if documents:
            # Create or update vector index with batch; embeddings are generated here
            if i == 0:
                # First batch: create new index
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            else:
                # Subsequent batches: add to existing index
                index.insert_nodes(documents)
            logger.info(f"Processed batch {i // batch_size + 1}: ingested {len(documents)} documents.")
        else:
            logger.warning(f"No documents in batch {i // batch_size + 1}.")

    logger.info(f"Ingested all documents into collection '{collection_name}'")


@cli.command()
@click.option("--query", required=True, help="Search query.")
@click.option("--collection-name", default="switch_scores", help="ChromaDB collection name.")
@click.option("--top-k", default=5, help="Number of top results.")
def search(query: str, collection_name: str, top_k: int) -> None:
    """
    Perform similarity search on the vector index.

    This command retrieves the most similar document chunks to the query
    based on vector similarity (cosine distance). Useful for finding
    switches with specific characteristics without LLM synthesis.
    """
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # Reconstruct index from persisted vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Create retriever for similarity search
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    for i, result in enumerate(results):
        # Log first 200 characters of each result for preview
        logger.info(f"Result {i + 1}: {cast(str, result.node.text)[:200]}...")


@cli.command()
@click.option("--query", required=True, help="Query for LLM response.")
@click.option("--collection-name", default="switch_scores", help="ChromaDB collection name.")
def query(query: str, collection_name: str) -> None:
    """
    Query the index with natural language and get LLM-synthesized response.

    This command uses retrieval-augmented generation (RAG):
    1. Retrieve relevant document chunks via similarity search
    2. Pass chunks to deepseek-r1 LLM for context-aware response generation
    3. Return synthesized answer based on switch data
    """
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Create QA template for initial answer generation
    # {context_str}: Retrieved document chunks relevant to the query
    # {query_str}: The user's original question
    qa_template = PromptTemplate(
        "You are an expert assistant specializing in mechanical keyboard switches. "
        "Based on the provided context from switch score sheets, answer the question accurately, "
        "comprehensively, and objectively. Include specific details like scores, comparisons, "
        "and manufacturer info when relevant. If the context doesn't fully answer the question, "
        "state that clearly.\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n\n"
        "Answer:"
    )

    # Create refinement template to improve initial answers
    # {query_str}: The user's original question (repeated for context)
    # {existing_answer}: The current answer being refined
    # {context_msg}: Additional retrieved context for refinement
    refine_template = PromptTemplate(
        "You are refining an answer about mechanical keyboard switches. "
        "Original question: {query_str}\n\n"
        "Existing answer: {existing_answer}\n\n"
        "New context: {context_msg}\n\n"
        "Refine the answer to be more accurate and detailed, incorporating the new context. "
        "Maintain objectivity and include specific scores or comparisons.\n\n"
        "Refined Answer:"
    )

    # Create response synthesizer with custom prompts
    response_synthesizer = CompactAndRefine(text_qa_template=qa_template, refine_template=refine_template)

    # Create query engine with custom synthesizer
    query_engine = index.as_query_engine(response_synthesizer=response_synthesizer)
    response = query_engine.query(query)
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    cli()
