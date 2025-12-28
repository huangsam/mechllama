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
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import chromadb
import click
import pandas as pd
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import TextNode
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
@click.option("--csv-path", default="datalake/switch-scores/1-Composite Overall Total Score Sheet.csv", help="Path to the CSV file with switch scores.")
@click.option("--collection-name", default="switch_scores", help="ChromaDB collection name.")
@click.option("--batch-size", default=10, help="Number of PDFs to process per batch.")
def ingest(data_dir: str, csv_path: str, collection_name: str, batch_size: int) -> None:
    """
    Ingest PDF files into ChromaDB vector index in batches, enriched with CSV scores.

    This command:
    1. Loads switch scores from the CSV file
    2. Connects to ChromaDB server running on localhost:8000
    3. Creates or retrieves a collection for storing vectors
    4. Loads PDF documents in batches, extracting text content
    5. Enriches each PDF document with matching scores from CSV
    6. Embeds the enriched text using bge-m3 via Ollama and adds to index incrementally
    7. Stores embeddings in ChromaDB for later retrieval

    Failed PDFs are logged as warnings and skipped. Batching helps with memory management.
    CSV enrichment ensures both qualitative descriptions and quantitative scores are available.
    """
    # Initialize ChromaDB HTTP client for server-mode persistence
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    try:
        chroma_client.heartbeat()  # Verify ChromaDB server is accessible
    except Exception as e:
        logger.error(f"ChromaDB not running: {e}")
        return

    # Load CSV scores for enrichment
    try:
        df = pd.read_csv(csv_path, skiprows=7, header=None)  # Skip header rows, no header
        # Manually set column names based on the CSV structure
        column_names = [
            "Rank",
            "Switch Name",
            "Date",
            "Manufacturer",
            "Type",
            "Push Feel",
            "Wobble",
            "Sound",
            "Context",
            "Other",
            "Timeless Total",
            "Time Wtd. Total",
            "",
            "",
            "Mfg Rank",
            "Manufacturer Name",
            "Switches Tested",
            "Mfg Push Feel",
            "Mfg Wobble",
            "Mfg Sound",
            "Mfg Context",
            "Mfg Other",
            "Mfg Timeless Total",
            "Mfg Time Wtd. Total",
        ]
        df.columns = column_names
        # Columns: Rank, Switch Name, Date, Manufacturer, Type, Push Feel, Wobble, Sound, Context, Other, Timeless Total, Time Wtd. Total
        score_columns = ["Push Feel", "Wobble", "Sound", "Context", "Other", "Timeless Total", "Time Wtd. Total"]
        score_dict = {}
        for _, row in df.iterrows():
            switch_name = str(row.get("Switch Name", "")).strip().lower()  # Normalize for matching
            if switch_name:
                scores = {}
                for col in score_columns:
                    val = row.get(col, "N/A")
                    # Convert numeric scores to float for ChromaDB filtering
                    if col in ["Push Feel", "Wobble", "Sound", "Context", "Other", "Timeless Total", "Time Wtd. Total"]:
                        try:
                            scores[col] = float(val) if val != "N/A" else "N/A"
                        except (ValueError, TypeError):
                            scores[col] = "N/A"
                    else:
                        scores[col] = val
                score_dict[switch_name] = scores
        logger.info(f"Loaded scores for {len(score_dict)} switches from {csv_path}")
    except Exception as e:
        logger.error(f"Failed to load CSV {csv_path}: {e}")
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

                # Enrich with CSV scores
                switch_name = Path(file_path).stem.strip().lower()  # Extract switch name from filename
                scores = score_dict.get(switch_name, {})
                enriched_docs = []
                for doc in docs:
                    # Create enriched text
                    score_text = "\n".join([f"{k}: {v}" for k, v in scores.items() if v != "N/A"])
                    enriched_text = doc.text
                    if score_text:
                        enriched_text += f"\n\nScores from CSV:\n{score_text}"
                    # Create new document with enriched text and metadata
                    from llama_index.core.schema import Document

                    enriched_doc = Document(text=enriched_text, metadata={**doc.metadata, **scores})
                    enriched_docs.append(enriched_doc)

                documents.extend(enriched_docs)
                logger.info(f"Loaded and enriched {file_path}")
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
@click.option("--collection-name", default="switch_scores_enriched_numeric", help="ChromaDB collection name.")
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

    # Apply metadata filtering for score-based queries
    where_clause = get_score_filter(query)

    # Create retriever for similarity search with optional filtering
    retriever = index.as_retriever(similarity_top_k=top_k, vector_store_kwargs={"where": where_clause} if where_clause else {})
    results = retriever.retrieve(query)
    for i, result in enumerate(results):
        # Log first 200 characters of each result for preview
        text_node = cast(TextNode, result.node)
        logger.info(f"Result {i + 1}: {cast(str, text_node.text)[:200]}...")


def get_score_filter(query: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata filter from query based on score ranges.

    Returns a ChromaDB where clause if filtering is detected, None otherwise.
    """
    # Configuration for score filtering - easily extensible
    SCORE_CONFIG = {
        "push feel": {
            "field": "Push Feel",
            "ranges": {
                "highest": {"$gte": 25},  # 25-35
                "mid": {"$and": [{"Push Feel": {"$gte": 15}}, {"Push Feel": {"$lte": 24}}]},  # 15-24
                "lowest": {"$lte": 14},  # 0-14
            },
            "keywords": ["push feel", "pushfeel"],
        },
        "wobble": {
            "field": "Wobble",
            "ranges": {
                "highest": {"$gte": 18},  # 18-25
                "mid": {"$and": [{"Wobble": {"$gte": 9}}, {"Wobble": {"$lte": 17}}]},  # 9-17
                "lowest": {"$lte": 8},  # 0-8
            },
            "keywords": ["wobble"],
        },
        "sound": {
            "field": "Sound",
            "ranges": {
                "highest": {"$gte": 8},  # 8-10
                "mid": {"$and": [{"Sound": {"$gte": 4}}, {"Sound": {"$lte": 7}}]},  # 4-7
                "lowest": {"$lte": 3},  # 0-3
            },
            "keywords": ["sound"],
        },
        "context": {
            "field": "Context",
            "ranges": {
                "highest": {"$gte": 14},  # 14-20
                "mid": {"$and": [{"Context": {"$gte": 7}}, {"Context": {"$lte": 13}}]},  # 7-13
                "lowest": {"$lte": 6},  # 0-6
            },
            "keywords": ["context"],
        },
        "other": {
            "field": "Other",
            "ranges": {
                "highest": {"$gte": 8},  # 8-10
                "mid": {"$and": [{"Other": {"$gte": 4}}, {"Other": {"$lte": 7}}]},  # 4-7
                "lowest": {"$lte": 3},  # 0-3
            },
            "keywords": ["other"],
        },
        # Add more score types here as needed
    }

    query_lower = query.lower()
    where_clause = None

    for _, config in SCORE_CONFIG.items():
        if any(keyword in query_lower for keyword in config["keywords"]):
            # Check for ranking adjectives
            if any(word in query_lower for word in ["highest", "best", "top"]):
                where_clause = {config["field"]: config["ranges"]["highest"]}
                break
            elif any(word in query_lower for word in ["mid", "medium", "middle", "average"]):
                where_clause = config["ranges"]["mid"]
                break
            elif any(word in query_lower for word in ["lowest", "worst", "bottom"]):
                where_clause = {config["field"]: config["ranges"]["lowest"]}
                break

    return where_clause


@cli.command()
@click.option("--query", required=True, help="Query for LLM response.")
@click.option("--collection-name", default="switch_scores_enriched_numeric", help="ChromaDB collection name.")
@click.option("--top-k", default=20, help="Number of documents to retrieve for context.")
def query(query: str, collection_name: str, top_k: int) -> None:
    """
    Query the index with natural language and get LLM-synthesized response.

    This command uses retrieval-augmented generation (RAG) with config-driven metadata filtering:
    1. Detects ranking queries using configurable patterns and applies score filters
    2. Retrieves relevant document chunks via similarity search
    3. Passes chunks to deepseek-r1 LLM for context-aware response generation
    4. Returns synthesized answer based on switch data
    """
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Apply metadata filtering for score-based queries
    where_clause = get_score_filter(query)

    # Create QA template for initial answer generation
    # {context_str}: Retrieved document chunks relevant to the query
    # {query_str}: The user's original question
    qa_template = PromptTemplate(
        "You are an expert assistant specializing in mechanical keyboard switches. "
        "Based on the provided context from switch score sheets, answer the question accurately, "
        "comprehensively, and objectively.\n\n"
        "IMPORTANT: Sound scores range from 0-10, where 10 is the best. Push Feel scores range from 0-35. "
        "When asked for 'highest' scores, identify the switches with the MAXIMUM numerical values. "
        "When asked for 'lowest' scores, identify the switches with the MINIMUM numerical values.\n\n"
        "Include specific details like exact scores, comparisons, and manufacturer info when relevant. "
        "If the context doesn't fully answer the question, state that clearly.\n\n"
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

    # Create query engine with custom synthesizer, increased retrieval count, and metadata filtering
    if where_clause:
        from llama_index.core.query_engine import RetrieverQueryEngine

        retriever = index.as_retriever(similarity_top_k=top_k, vector_store_kwargs={"where": where_clause})
        query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
    else:
        query_engine = index.as_query_engine(response_synthesizer=response_synthesizer, similarity_top_k=top_k)
    response = query_engine.query(query)
    logger.info(f"Response: {response}")


if __name__ == "__main__":
    cli()
