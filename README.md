# Mechanical Keyboard Switch Analysis CLI

Ask questions about mechanical keyboard switches and get AI-powered answers based on comprehensive reviews and scores.

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai): `ollama pull bge-m3` and `ollama pull deepseek-r1`
- ChromaDB: `docker run -p 8000:8000 chromadb/chroma:latest`

### Install & Setup
```bash
git clone <repository-url>
cd mechllama
uv sync

# Optional: Configure environment variables
cp .env.example .env
# Edit .env with your preferred settings (never commit .env to version control)
```

## Configuration

The application supports configuration via environment variables. Copy `.env.example` to `.env` and modify as needed:

- **OLLAMA_BASE_URL**: Ollama server URL (default: http://localhost:11434)
- **OLLAMA_EMBEDDING_MODEL**: Embedding model name (default: bge-m3)
- **OLLAMA_LLM_MODEL**: LLM model name (default: deepseek-r1:latest)
- **OLLAMA_TIMEOUT**: Request timeout in seconds (default: 120.0)
- **OLLAMA_CONTEXT_WINDOW**: Context window size (default: 128000)
- **CHROMA_HOST**: ChromaDB server host (default: localhost)
- **CHROMA_PORT**: ChromaDB server port (default: 8000)

### Setup Data
```bash
uv run python main.py ingest  # Index switch data
```

## Usage

### Ask Questions
```bash
uv run python main.py query --query "Which switches have the highest sound scores?"
```

### Search Raw Content
```bash
uv run python main.py search --query "linear switches with good sound"
```

## Intelligent Filtering

Automatically understands score-based questions:

### Score Categories
- **Push Feel** (0-35): Actuation feel
- **Wobble** (0-25): Stem stability
- **Sound** (0-10): Acoustic performance
- **Context** (0-20): Overall suitability
- **Other** (0-10): Miscellaneous factors

### Natural Language Queries

```bash
# Performance tiers
uv run python main.py query --query "switches with highest sound scores"
uv run python main.py query --query "best push feel switches"
uv run python main.py query --query "worst wobble switches"

# Combined criteria
uv run python main.py query --query "linear switches with high sound but low wobble"

# General questions
uv run python main.py query --query "Tell me about tactile switches"
uv run python main.py query --query "Compare Cherry MX vs Alpacas"
```

## Examples

### Recommendations
```bash
uv run python main.py query --query "Which switches for gaming?"
uv run python main.py query --query "Best switches for typing?"
```

### Analysis
```bash
uv run python main.py query --query "Compare top 3 linear switches by sound"
uv run python main.py query --query "Switches with excellent feel but quiet sound"
```

## Data Sources

- **PDF Reviews**: Detailed subjective evaluations from switch-scores repository
- **CSV Scores**: Quantitative metrics across 5 categories
- **Combined Analysis**: Links descriptions with numerical scores

## Command Reference

### Ingest Data
```bash
uv run python main.py ingest [OPTIONS]
# --data-dir PATH: PDF directory (default: datalake/switch-scores)
# --csv-path PATH: Score CSV file path
# --batch-size INT: PDFs per batch (default: 10)
```

### Query with AI
```bash
uv run python main.py query --query "YOUR QUESTION" [OPTIONS]
# --collection-name TEXT: Collection name (default: switch_scores_enriched_numeric)
# --top-k INT: Documents to retrieve (default: 20)
```

### Search Documents
```bash
uv run python main.py search --query "SEARCH TERM" [OPTIONS]
```

---

**For developers**: See [AGENTS.md](AGENTS.md) for technical details.
