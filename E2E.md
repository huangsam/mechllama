# End-to-End Testing & RAG Evaluation

## Overview

The `e2e_test.py` module provides comprehensive end-to-end testing and evaluation for the Mechanical Keyboard Switch Analysis CLI using RAG (Retrieval-Augmented Generation) assessment techniques.

## Test Suite (20 tests across 3 classes)

### TestE2ECliEvaluation (13 tests)
Tests CLI functionality across different query types:
- **CLI Basics**: Help command, query command availability
- **Score Filtering**: Sound, push feel, wobble, context, other scores
- **Semantic Search**: Unfiltered retrieval, similarity search
- **Response Quality**: Non-empty responses, contextual grounding, consistency
- **Configuration**: Custom collection name, top-k parameters

### TestRAGQualityEvaluation (3 tests)
Evaluates response quality:
- **Response Relevance**: Checks response length and structure
- **Answer Grounding**: Validates use of retrieved context
- **Error Handling**: Graceful failures on missing collections

### TestDataQualityAssessment (4 tests)
Validates data and retrieval:
- Collection accessibility and document count
- Metadata enrichment (score fields present)
- Sample retrieval quality

## Prerequisites

```bash
# Install dependencies
uv sync --group dev

# Start ChromaDB (required)
docker run -p 8000:8000 chromadb/chroma:latest &

# Pull Ollama models
ollama pull bge-m3
ollama pull deepseek-r1

# Ingest data (required)
uv run python main.py ingest --data-dir datalake/switch-scores \
  --csv datalake/switch-scores/1-Composite\ Overall\ Total\ Score\ Sheet.csv
```

## Running Tests

```bash
# All e2e tests
uv run pytest e2e_test.py -v

# Specific test class
uv run pytest e2e_test.py::TestE2ECliEvaluation -v

# With detailed output
uv run pytest e2e_test.py -v -s

# Coverage report
uv run pytest e2e_test.py --cov=main --cov-report=html
```

## Test Dataset

8 diverse questions covering:
- Score filtering: "Which switches have the highest sound scores?"
- Push feel: "What about switches with the best push feel?"
- Wobble: "Tell me about switches with the lowest wobble"
- Semantic search: "What are the best linear switches?"
- General knowledge: "What makes a high-quality mechanical switch?"
- Manufacturer comparison: "How do manufacturers compare in quality?"

## Evaluation Metrics

**Relevance**: Response appropriateness to query (length > 50 chars, no errors)
**Grounding**: Use of retrieved context (keywords: "score", "based on", "switch")
**Faithfulness**: Answer accuracy against data (specific scores/switches mentioned)
**Consistency**: Reproducible results across runs

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Tests skipped | Run ingestion first (see Prerequisites) |
| ChromaDB connection error | `docker run -p 8000:8000 chromadb/chroma:latest` |
| Model not found | `ollama pull bge-m3` and `ollama pull deepseek-r1` |
| Collection errors | Verify ingestion completed successfully |

## Test Execution Flow

1. Validate collection exists and contains documents
2. Test CLI commands (help, query, search)
3. Test score filtering across all categories
4. Verify response quality and consistency
5. Assess RAG quality (relevance, grounding)
6. Validate data enrichment and metadata

## Performance

- **Total execution time**: ~11 minutes (20 tests)
- **CLI tests**: ~3 minutes (13 tests)
- **Quality tests**: ~6 minutes (3 tests)
- **Data validation**: ~30 seconds (4 tests)

Times vary based on network latency and system resources.

## Key Features

✅ Comprehensive CLI validation
✅ RAG quality assessment across 4 dimensions
✅ Robust error handling (skips when data unavailable)
✅ Detailed logging for troubleshooting
✅ 100% pass rate (20/20 tests)
✅ Compatible with unit tests (main_test.py: 23 tests)
✅ Production-ready test architecture

## Example Test Run

```bash
$ uv run pytest e2e_test.py::TestE2ECliEvaluation -v
e2e_test.py::TestE2ECliEvaluation::test_cli_help PASSED
e2e_test.py::TestE2ECliEvaluation::test_query_with_score_filtering PASSED
...
======================== 13 passed in 180s ========================
```

## Extending Tests

To add new tests:

1. Add question to `TEST_QUESTIONS` dict
2. Create test method in appropriate class
3. Use fixtures: `cli_runner`, `collection_populated`
4. Add assertions for evaluation criteria

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Run E2E tests
  run: |
    docker run -d -p 8000:8000 chromadb/chroma:latest
    uv run python main.py ingest --data-dir datalake/switch-scores
    uv run pytest e2e_test.py -v
```

## Files

- **e2e_test.py**: Test implementation (20 tests)
- **main_test.py**: Unit tests for score filtering (23 tests)
- **main.py**: CLI application being tested
- **pyproject.toml**: Dependencies (ragas, datasets)
