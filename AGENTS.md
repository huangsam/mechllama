# Developer Guide - Mechanical Keyboard Switch Analysis CLI

This document contains technical details, development setup, testing, and configuration for contributors and maintainers.

## Architecture

### Core Components
- **CLI Framework**: Click for command-line interface with three main commands
- **Vector Search**: ChromaDB HTTP client for persistent vector storage
- **Embeddings**: Ollama bge-m3 (1024 dimensions, multilingual) for semantic search
- **LLM**: Ollama deepseek-r1 (128K context window) for natural language responses
- **Response Synthesis**: CompactAndRefine with custom prompts optimized for switch analysis
- **Data Processing**: Pandas for CSV parsing, batch processing for memory efficiency

### Data Flow
1. **Ingestion**: PDF text extraction → CSV score enrichment → Embedding generation → ChromaDB storage
2. **Query Processing**: Natural language input → Score filter detection → Metadata filtering → Vector search → LLM synthesis
3. **Response Generation**: Retrieved context + custom prompts → AI reasoning → Structured answers

### Key Files
- `main.py`: Main CLI application with LlamaIndex integration
- `main_test.py`: Comprehensive test suite (23 tests covering filtering logic)
- `pyproject.toml`: Project configuration with uv dependency management
- `datalake/switch-scores/`: Data directory with PDFs and CSV scores

## Development Setup

### Prerequisites
- Python 3.11+
- uv package manager
- Ollama with bge-m3 and deepseek-r1 models
- ChromaDB server running locally

### Installation
```bash
git clone <repository-url>
cd mechllama
uv sync --group dev  # Includes testing and linting tools
```

### Development Workflow
```bash
# Run tests
uv run pytest main_test.py -v

# Format code
uv run ruff format

# Lint code
uv run ruff check --fix

# Type checking
uv run mypy main.py
```

## Testing

### Test Coverage
The test suite covers:
- Score filtering logic for all 5 categories (push feel, wobble, sound, context, other)
- Keyword variations and synonyms (highest/best/top, mid/medium/average, lowest/worst/bottom)
- Case insensitivity handling
- Combined query processing
- Edge cases and error conditions
- Configuration order independence

### Running Tests
```bash
# All tests
uv run pytest main_test.py -v

# Specific test class
uv run pytest main_test.py::TestGetScoreFilter -v

# Coverage report
uv run pytest --cov=main --cov-report=html
```

### Test Structure
- **23 total tests** across filtering scenarios
- **100% pass rate** required for all changes
- Tests validate both filtering logic and natural language processing

## Configuration

### Environment Variables

The application uses python-dotenv to load configuration from a `.env` file. All configuration is now environment-variable driven for better deployment flexibility:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434          # Ollama server URL
OLLAMA_EMBEDDING_MODEL=bge-m3                   # Embedding model name
OLLAMA_LLM_MODEL=deepseek-r1:latest            # LLM model name
OLLAMA_TIMEOUT=120.0                           # Request timeout in seconds
OLLAMA_CONTEXT_WINDOW=128000                   # Context window size

# ChromaDB Configuration
CHROMA_HOST=localhost                          # ChromaDB server host
CHROMA_PORT=8000                               # ChromaDB server port
```

### LlamaIndex Settings

Settings are now loaded from environment variables with sensible defaults:

```python
# In main.py - now configurable via environment variables
Settings.embed_model = OllamaEmbedding(
    model_name=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)
Settings.llm = Ollama(
    model=os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:latest"),
    request_timeout=float(os.getenv("OLLAMA_TIMEOUT", "120.0")),
    context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "128000"))
)
```

### Score Filtering Configuration
```python
SCORE_CONFIG = {
    "sound": {"highest": 7.5, "mid": (5.0, 7.4), "lowest": 4.9},
    "push_feel": {"highest": 7.5, "mid": (5.0, 7.4), "lowest": 4.9},
    # ... other categories
}
```
Ranges derived empirically from score distribution analysis.

### Collection Names
- **Ingestion**: `switch_scores` (default)
- **Query**: `switch_scores_enriched_numeric` (default)

## Troubleshooting

### Common Development Issues

**ChromaDB Connection Error**
```bash
# Check if server is running
docker ps | grep chroma

# Start ChromaDB server
docker run -p 8000:8000 chromadb/chroma:latest

# Or run locally
pip install chromadb
chroma run --host 0.0.0.0 --port 8000
```

**Ollama Model Issues**
```bash
# Pull required models
ollama pull bge-m3
ollama pull deepseek-r1

# Check available models
ollama list

# Test model loading
ollama run deepseek-r1 "Hello"
```

**Memory Issues During Development**
- Reduce batch size for ingestion: `--batch-size 5`
- Process fewer files at once
- Monitor memory usage during embedding generation
- Consider using smaller embedding models for development

**Test Failures**
- Ensure test data matches expected formats
- Check score filtering logic for edge cases
- Verify natural language processing handles variations correctly
- Run individual tests to isolate issues

### Debugging Tips
- Enable verbose logging: Set logging level to DEBUG
- Check ChromaDB collections: Use ChromaDB admin interface
- Validate embeddings: Test similarity search manually
- Monitor LLM responses: Check prompt templates and context

## Performance Optimization

### Ingestion Performance
- **Batch Processing**: Default 10 PDFs per batch, adjustable via `--batch-size`
- **Memory Management**: Process files incrementally to avoid OOM errors
- **Embedding Caching**: ChromaDB handles persistent storage across sessions

### Query Performance
- **Metadata Filtering**: Pre-filters results before vector search for score queries
- **Top-K Limiting**: Default 20 documents, adjustable for precision vs speed
- **Context Window**: 128K tokens available for comprehensive analysis

### Monitoring
- Ingestion progress logged with file counts and batch status
- Query performance tracked via response times
- Error handling with graceful degradation

## Contributing

### Code Standards
- **Formatting**: ruff format (Black-compatible)
- **Linting**: ruff check with custom rules
- **Type Hints**: Full typing coverage required
- **Documentation**: Comprehensive docstrings and inline comments

### Pull Request Process
1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite: `uv run pytest`
4. Format and lint: `uv run ruff format && uv run ruff check --fix`
5. Update documentation if needed
6. Submit PR with description of changes

### Adding New Features
- **Score Categories**: Update SCORE_CONFIG and add tests
- **Query Types**: Extend get_score_filter() with new patterns
- **Data Sources**: Modify ingestion logic for new file types
- **Models**: Update Settings for different Ollama models

## Deployment

### Production Setup
- Use production ChromaDB instance (not localhost)
- Configure Ollama with GPU acceleration if available
- Set appropriate timeouts for production workloads
- Monitor memory usage and scale accordingly

## Deployment

### Production Setup
- Use production ChromaDB instance (not localhost)
- Configure Ollama with GPU acceleration if available
- Set appropriate timeouts for production workloads
- Monitor memory usage and scale accordingly

### Environment Variables
```bash
# Optional environment configuration
export CHROMA_HOST=production-chroma.example.com
export OLLAMA_BASE_URL=http://production-ollama:11434
# ... other variables as needed
```
