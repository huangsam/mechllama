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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import chromadb
import click
import pandas as pd
from dotenv import load_dotenv
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

# Load environment variables from .env file
load_dotenv()

# Global LlamaIndex settings for embeddings and LLM
# Using Ollama for local, private processing without external API calls
# bge-m3: High-quality multilingual embedding model (1024 dimensions) optimized for text retrieval
# deepseek-r1: Advanced reasoning model with 128K context window for comprehensive analysis
# 120s timeout: Allows sufficient time for complex reasoning tasks
Settings.embed_model = OllamaEmbedding(
    model_name=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)
Settings.llm = Ollama(
    model=os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:latest"),
    request_timeout=float(os.getenv("OLLAMA_TIMEOUT", "120.0")),
    context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "128000"))
)


def get_chroma_client() -> chromadb.HttpClient:
    """
    Create and return a ChromaDB HTTP client with configurable host and port.

    Returns:
        Configured ChromaDB HttpClient instance
    """
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    return chromadb.HttpClient(host=host, port=port)


@click.group()
def cli() -> None:
    """
    Mechanical Keyboard Switch Analysis CLI

    A comprehensive tool for indexing and querying mechanical keyboard switch scores
    using LlamaIndex RAG (Retrieval-Augmented Generation) with local AI models.

    Commands:
      ingest    Index PDF switch reviews enriched with CSV scores into vector database
      search    Find switches by semantic similarity (returns raw document snippets)
      query     Ask natural language questions (returns AI-synthesized answers)

    All commands support metadata filtering for score-based queries (highest/mid/lowest
    sound, push feel, wobble, context, and other scores).
    """
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
    chroma_client = get_chroma_client()
    try:
        chroma_client.heartbeat()  # Verify ChromaDB server is accessible
    except Exception as e:
        logger.error(f"ChromaDB not running: {e}")
        return

    # Load CSV scores for enrichment
    try:
        # CSV has 7 header rows before data starts, so skip them
        df = pd.read_csv(csv_path, skiprows=7, header=None)
        # Manually set column names based on the CSV structure from switch-scores repo
        # This CSV contains comprehensive switch evaluation data with multiple score categories
        column_names = [
            "Rank",  # Overall ranking position
            "Switch Name",  # Switch model identifier
            "Date",  # Review publication date
            "Manufacturer",  # Company that makes the switch
            "Type",  # Switch type (linear, tactile, clicky)
            "Push Feel",  # Actuation feel score (0-35)
            "Wobble",  # Stem stability score (0-25)
            "Sound",  # Acoustic performance score (0-10)
            "Context",  # Overall context/suitability score (0-20)
            "Other",  # Miscellaneous factors score (0-10)
            "Timeless Total",  # Time-weighted total score
            "Time Wtd. Total",  # Alternative time-weighted calculation
            "",
            "",  # Empty columns
            "Mfg Rank",  # Manufacturer ranking
            "Manufacturer Name",  # Manufacturer name (duplicate)
            "Switches Tested",  # Number of switches tested by manufacturer
            "Mfg Push Feel",  # Manufacturer average push feel
            "Mfg Wobble",  # Manufacturer average wobble
            "Mfg Sound",  # Manufacturer average sound
            "Mfg Context",  # Manufacturer average context
            "Mfg Other",  # Manufacturer average other
            "Mfg Timeless Total",  # Manufacturer average timeless total
            "Mfg Time Wtd. Total",  # Manufacturer average time-weighted total
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
    chroma_client = get_chroma_client()
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

    Analyzes natural language queries to detect ranking requests (highest/mid/lowest)
    and applies ChromaDB metadata filters accordingly. Supports all major switch
    evaluation categories with empirically-derived score ranges.

    Score Range Rationale:
    - Highest: Top ~25% of possible scores (prioritizes exceptional switches)
    - Mid: Middle ~50% of possible scores (balanced performance range)
    - Lowest: Bottom ~25% of possible scores (identifies poor performers)

    Returns a ChromaDB where clause if filtering is detected, None otherwise.
    """
    # Configuration for score filtering - easily extensible
    # Score ranges are derived from the switch-scores dataset analysis
    # Each category has distinct scales and meaningful threshold boundaries
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
    Query the index with natural language and get AI-synthesized response.

    This command implements retrieval-augmented generation (RAG) with intelligent
    metadata filtering for analytical queries:

    1. Detects ranking queries (highest/mid/lowest + score category) and applies
       ChromaDB metadata filters to pre-filter relevant documents
    2. Retrieves document chunks via vector similarity search (filtered or unfiltered)
    3. Passes chunks to deepseek-r1 LLM with custom prompts for context-aware synthesis
    4. Returns comprehensive, objective answers with specific scores and comparisons

    For score ranking queries, only switches meeting the criteria are considered.
    For general queries, full collection search is performed without filtering.

    Examples:
      "Which switches have the highest sound scores?" → Filtered search
      "Tell me about linear switches" → Unfiltered search
    """
    chroma_client = get_chroma_client()
    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Apply metadata filtering for score-based queries
    where_clause = get_score_filter(query)

    # Create custom prompt templates optimized for switch analysis
    # QA template: Guides initial answer generation with scoring context and objectivity requirements
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

    # Refinement template: Improves answer quality by incorporating additional context
    # Used by CompactAndRefine synthesizer to iteratively enhance responses
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

    # Create query engine with conditional filtering behavior:
    # - If score filtering detected: Use RetrieverQueryEngine with metadata pre-filtering
    # - If no filtering needed: Use standard query engine for full collection search
    # This optimizes performance by reducing context size for analytical queries
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
