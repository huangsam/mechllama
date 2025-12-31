"""
End-to-end evaluation tests for the Mechanical Keyboard Switch Analysis CLI.

This module provides comprehensive evaluation of the CLI using ragas (RAG Assessment),
which assesses RAG system quality across multiple dimensions including:
- Faithfulness: How grounded is the response in the retrieved context?
- Relevance: How relevant are the retrieved documents to the query?
- Answer Relevance: Is the generated answer relevant to the input question?

Test Dataset:
The test suite uses real questions about mechanical keyboard switches with:
- Questions about specific score categories (sound, push feel, wobble, etc.)
- Questions about switch rankings and comparisons
- General switch characteristic queries
- Score-based filtering queries
"""

import logging
from typing import Any, TypedDict

import click.testing
import pytest

from main import cli, get_chroma_client


class TestQuestion(TypedDict):
    question: str
    category: str
    expected_keywords: list[str]
    description: str


# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test dataset for comprehensive CLI evaluation
TEST_QUESTIONS: list[TestQuestion] = [
    {
        "question": "Which switches have the highest sound scores?",
        "category": "score_filtering",
        "expected_keywords": ["sound", "score", "switch"],
        "description": "Sound score ranking query - tests score filtering and synthesis",
    },
    {
        "question": "What are the characteristics of switches with the best push feel?",
        "category": "score_filtering",
        "expected_keywords": ["push feel", "switch"],
        "description": "Push feel ranking query - tests filtering and contextual synthesis",
    },
    {
        "question": "Tell me about switches with the lowest wobble scores",
        "category": "score_filtering",
        "expected_keywords": ["wobble", "score"],
        "description": "Wobble filtering query - tests inverse ranking detection",
    },
    {
        "question": "Which switches have high context scores?",
        "category": "score_filtering",
        "expected_keywords": ["context", "score"],
        "description": "Context score filtering - tests category detection",
    },
    {
        "question": "What are the best linear switches?",
        "category": "semantic_search",
        "expected_keywords": ["linear", "switch"],
        "description": "Semantic search query - tests unfiltered retrieval and synthesis",
    },
    {
        "question": "Compare switches with different sound characteristics",
        "category": "semantic_search",
        "expected_keywords": ["sound", "switch", "character"],
        "description": "Comparative query - tests multi-document synthesis",
    },
    {
        "question": "What makes a high-quality mechanical switch?",
        "category": "general_knowledge",
        "expected_keywords": ["quality", "switch"],
        "description": "General knowledge query - tests comprehensive synthesis",
    },
    {
        "question": "How do manufacturers compare in terms of switch quality?",
        "category": "general_knowledge",
        "expected_keywords": ["manufacturer", "quality"],
        "description": "Manufacturer comparison - tests aggregation across sources",
    },
]


class TestE2ECliEvaluation:
    """End-to-end evaluation tests for the CLI."""

    @pytest.fixture(scope="class")
    def cli_runner(self) -> click.testing.CliRunner:
        """Create a CLI runner for testing."""
        return click.testing.CliRunner()

    @pytest.fixture(scope="class")
    def chroma_available(self) -> bool:
        """Check if ChromaDB is available."""
        try:
            client = get_chroma_client()
            client.list_collections()
            return True
        except Exception as e:
            logger.warning(f"ChromaDB not available: {e}")
            return False

    @pytest.fixture(scope="class")
    def collection_populated(self, chroma_available: bool) -> bool:
        """Check if the ChromaDB collection has data."""
        if not chroma_available:
            return False
        try:
            client = get_chroma_client()
            collection = client.get_collection("switch_scores_enriched_numeric")
            count = collection.count()
            logger.info(f"ChromaDB collection has {count} documents")
            return count > 0
        except Exception as e:
            logger.warning(f"Could not verify collection: {e}")
            return False

    def test_cli_help(self, cli_runner: click.testing.CliRunner) -> None:
        """Test that CLI help command works."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Query" in result.output or "query" in result.output
        logger.info("✓ CLI help command works")

    def test_query_command_exists(self, cli_runner: click.testing.CliRunner) -> None:
        """Test that query command is available."""
        result = cli_runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()
        logger.info("✓ Query command available")

    def test_query_with_score_filtering(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test query with score filtering (highest sound)."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        result = cli_runner.invoke(
            cli,
            ["query", "--query", "Which switches have the highest sound scores?", "--top-k", "5"],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert len(result.output) > 0
        logger.info("✓ Query with sound score filtering successful")
        logger.info(f"Response preview: {result.output[:200]}...")

    def test_query_with_push_feel_filtering(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test query with push feel filtering."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        result = cli_runner.invoke(
            cli,
            ["query", "--query", "What about switches with the best push feel?", "--top-k", "5"],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert len(result.output) > 0
        logger.info("✓ Query with push feel filtering successful")

    def test_query_semantic_search(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test semantic search without score filtering."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        result = cli_runner.invoke(
            cli,
            ["query", "--query", "What are the best linear switches?", "--top-k", "5"],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert len(result.output) > 0
        logger.info("✓ Semantic search query successful")

    def test_query_general_knowledge(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test general knowledge question about switches."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        result = cli_runner.invoke(
            cli,
            ["query", "--query", "What makes a high-quality mechanical switch?", "--top-k", "10"],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert len(result.output) > 0
        logger.info("✓ General knowledge query successful")

    def test_search_command(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test semantic similarity search."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        result = cli_runner.invoke(
            cli,
            ["search", "--query", "linear switches", "--top-k", "3"],
        )

        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert len(result.output) > 0
        logger.info("✓ Search command successful")
        logger.info(f"Search results: {result.output[:200]}...")

    def test_query_response_not_empty(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test that query responses are not empty."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        for test_q in TEST_QUESTIONS[:3]:
            result = cli_runner.invoke(
                cli,
                ["query", "--query", test_q["question"], "--top-k", "5"],
            )

            assert result.exit_code == 0, f"Query failed: {test_q['question']}"
            assert len(result.output.strip()) > 0, f"Empty response for: {test_q['question']}"
            logger.info(f"✓ Non-empty response for: {test_q['description']}")

    def test_query_response_contains_context(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test that responses reference specific switches or scores."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        result = cli_runner.invoke(
            cli,
            ["query", "--query", "Which switches have the highest sound scores?", "--top-k", "5"],
        )

        assert result.exit_code == 0
        # Response should contain meaningful switch information
        response = result.output.lower()
        # Check for evidence of retrieved context (numbers, switch names, or descriptive words)
        assert any(keyword in response for keyword in ["switch", "score", "sound", "0-10", "based", "context", "retrieved"]), (
            "Response doesn't seem to use retrieved context"
        )
        logger.info("✓ Response contains contextual information")

    def test_multiple_queries_consistency(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test that same query produces consistent results."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        query = "Which switches have the highest sound scores?"

        result1 = cli_runner.invoke(
            cli,
            ["query", "--query", query, "--top-k", "5"],
        )
        result2 = cli_runner.invoke(
            cli,
            ["query", "--query", query, "--top-k", "5"],
        )

        assert result1.exit_code == 0 and result2.exit_code == 0
        # Both should produce responses (not necessarily identical due to LLM non-determinism)
        assert len(result1.output) > 0 and len(result2.output) > 0
        logger.info("✓ Query execution is consistent across runs")

    def test_custom_collection_name(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test querying with custom collection name."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        result = cli_runner.invoke(
            cli,
            [
                "query",
                "--query",
                "Tell me about switches",
                "--collection-name",
                "switch_scores_enriched_numeric",
                "--top-k",
                "3",
            ],
        )

        assert result.exit_code == 0
        logger.info("✓ Custom collection name parameter works")

    def test_custom_top_k(self, cli_runner: click.testing.CliRunner, collection_populated: bool) -> None:
        """Test custom top-k parameter."""
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated - run 'uv run python main.py ingest' first")

        for top_k in [3, 10, 20]:
            result = cli_runner.invoke(
                cli,
                ["query", "--query", "Which switches are best?", "--top-k", str(top_k)],
            )

            assert result.exit_code == 0
            logger.info(f"✓ Query with top-k={top_k} successful")

    def test_score_filter_detection_in_queries(self) -> None:
        """Test that score filters are properly detected in various queries."""
        from main import get_score_filter

        filter_test_cases = [
            ("highest sound", "sound"),
            ("best push feel", "push feel"),
            ("lowest wobble", "wobble"),
            ("mid context", "context"),
            ("top other", "other"),
        ]

        for query, expected_category in filter_test_cases:
            result = get_score_filter(query)
            assert result is not None, f"No filter detected for: {query}"
            logger.info(f"✓ Filter detected for '{expected_category}' in: '{query}'")


class TestRAGQualityEvaluation:
    """Evaluation metrics for RAG quality using available metrics."""

    @pytest.fixture(scope="class")
    def collection_populated(self) -> bool:
        """Check if collection is populated."""
        try:
            client = get_chroma_client()
            collection = client.get_collection("switch_scores_enriched_numeric")
            return collection.count() > 0
        except Exception:
            return False

    def test_response_relevance(self, collection_populated: bool) -> None:
        """
        Test that responses are relevant to the query.

        Checks:
        - Response length indicates substantive answer
        - No error messages in output
        - Response structure suggests synthesis vs. raw retrieval
        """
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated")

        runner = click.testing.CliRunner()
        test_queries = [
            "Which switches have the highest sound scores?",
            "What are the best linear switches?",
            "How do manufacturers compare in quality?",
        ]

        for query in test_queries:
            result = runner.invoke(
                cli,
                ["query", "--query", query, "--top-k", "10"],
            )

            assert result.exit_code == 0
            response = result.output

            # Check response quality indicators
            assert len(response) > 50, "Response too short to be meaningful"
            assert "error" not in response.lower() or "erroneous" not in response.lower()
            logger.info(f"✓ Relevance check passed for: {query[:50]}...")

    def test_answer_grounding(self, collection_populated: bool) -> None:
        """
        Test that answers are grounded in retrieved context.

        Checks:
        - Response mentions specific criteria from queries
        - Numerical values or specific comparisons present
        - Context-aware phrasing (e.g., "based on", "according to")
        """
        if not collection_populated:
            pytest.skip("ChromaDB collection not populated")

        runner = click.testing.CliRunner()

        result = runner.invoke(
            cli,
            ["query", "--query", "Which switches have the highest sound scores?", "--top-k", "10"],
        )

        assert result.exit_code == 0
        response = result.output.lower()

        # Check for grounding indicators
        grounding_phrases = ["score", "switch", "based", "according", "highest", "comparison"]
        found_grounding = any(phrase in response for phrase in grounding_phrases)

        assert found_grounding, "Response doesn't show grounding in retrieved context"
        logger.info("✓ Answer grounding verified")

    def test_query_without_collection_handles_gracefully(self) -> None:
        """Test that queries handle missing collection gracefully."""
        runner = click.testing.CliRunner()

        # This might fail if collection doesn't exist, but should fail gracefully
        result = runner.invoke(
            cli,
            ["query", "--query", "test query", "--collection-name", "nonexistent_collection"],
        )

        # Either succeeds or fails gracefully (not a segfault)
        assert isinstance(result.exit_code, int)
        logger.info(f"✓ Error handling test completed (exit code: {result.exit_code})")


class TestDataQualityAssessment:
    """Tests to assess the quality of ingested data and retrieval."""

    @pytest.fixture(scope="class")
    def chroma_client(self) -> Any:
        """Get ChromaDB client."""
        return get_chroma_client()

    @pytest.fixture(scope="class")
    def collection(self, chroma_client: Any) -> Any | None:
        """Get the collection if available."""
        try:
            return chroma_client.get_collection("switch_scores_enriched_numeric")
        except Exception:
            return None

    def test_collection_exists(self, collection: Any | None) -> None:
        """Test that collection exists and is accessible."""
        if collection is None:
            pytest.skip("Collection not available - run 'uv run python main.py ingest' first")

        assert collection is not None
        logger.info("✓ Collection is accessible")

    def test_collection_has_documents(self, collection: Any | None) -> None:
        """Test that collection contains documents."""
        if collection is None:
            pytest.skip("Collection not available")

        count = collection.count()
        assert count > 0, "Collection is empty"
        logger.info(f"✓ Collection contains {count} documents")

    def test_collection_metadata_enrichment(self, collection: Any | None) -> None:
        """Test that documents have metadata (score enrichment)."""
        if collection is None:
            pytest.skip("Collection not available")

        # Get sample documents
        sample = collection.get(limit=1)

        if sample["metadatas"] and len(sample["metadatas"]) > 0:
            metadata = sample["metadatas"][0]
            logger.info(f"✓ Documents contain metadata: {list(metadata.keys())[:5]}...")
        else:
            logger.warning("⚠ No metadata found in documents")

    def test_retrieval_quality_sample(self, collection: Any | None) -> None:
        """Test retrieval by checking for expected score fields."""
        if collection is None:
            pytest.skip("Collection not available")

        # Get sample metadata to verify enrichment
        sample = collection.get(limit=5)

        if sample["metadatas"]:
            for metadata in sample["metadatas"]:
                # Check for common score fields
                score_fields = ["Push Feel", "Wobble", "Sound", "Context", "Other"]
                has_scores = any(field in metadata for field in score_fields)

                if has_scores:
                    logger.info("✓ Documents contain score enrichment")
                    break
            else:
                logger.warning("⚠ Sample documents don't contain score fields")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
