# SignalMesh Local Agent

A multi-agent research operations system component that provides intelligent workspace analysis, semantic search, and RAG-based indexing capabilities.

## Overview

The SignalMesh Local Agent is designed to scan, parse, and index files in a workspace, enabling semantic search and intelligent analysis of research data, code, and documentation. It combines file system monitoring, content parsing, and vector-based retrieval to provide powerful workspace intelligence.

## Features

- **Intelligent File Scanning**: Automatically discovers and tracks files in your workspace with modification date filtering
- **Multi-Format Parsing**: Extracts structured information from Python code, CSV data, Jupyter notebooks, and Markdown documents
- **Semantic Search**: RAG-based vector search using Qdrant and sentence transformers
- **Local-First**: No cloud dependencies - runs entirely on your local machine
- **Production-Ready**: Complete type hints, error handling, logging, and testing
- **Reply Agent Ready**: Comprehensive JSON output with context, insights, and suggestions for automated email drafting

## For Integrators (Reply Agent Team)

📘 **[Integration Guide](INTEGRATION_GUIDE.md)** - Complete guide for integrating with Reply Agent
📄 **[Quick Reference](docs/QUICK_REFERENCE.md)** - Essential fields and code snippets
📋 **[JSON Schema](docs/output_schema_v1.0.json)** - Output schema v1.0 for validation

The Local Agent outputs comprehensive JSON with:
- ✅ Original requester context (name, email, question, channel)
- ✅ Detailed file analysis with insights and highlights
- ✅ Suggested actions for follow-up
- ✅ Everything needed to draft personalized email responses

See the [Integration Guide](INTEGRATION_GUIDE.md) for details.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LocalAgent                             │
│  (Main Orchestrator - Coordinates all operations)           │
└──────────────┬──────────────┬────────────┬─────────────────┘
               │              │            │
     ┌─────────▼────┐  ┌─────▼──────┐  ┌──▼──────────┐
     │ FileScanner  │  │  Content   │  │ RAGManager  │
     │              │  │  Parser    │  │             │
     │ - Workspace  │  │            │  │ - Qdrant    │
     │   scanning   │  │ - Python   │  │ - Vector    │
     │ - Date       │  │ - CSV      │  │   search    │
     │   filtering  │  │ - Notebook │  │ - Embedding │
     │ - Type       │  │ - Markdown │  │             │
     │   classification│ │            │  │             │
     └──────────────┘  └────────────┘  └─────────────┘
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd signalmesh-local-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import asyncio
from datetime import datetime, timedelta
from src.agent import LocalAgent

# Initialize the agent
agent = LocalAgent(workspace_path="/path/to/workspace")

# Create a task to scan all files
task_ticket = {
    "task_id": "task_001",
    "workspace_path": "/path/to/workspace",
    "scan_all": True,
    "reindex": True
}

# Process the task
result = asyncio.run(agent.handle_task(task_ticket))

print(f"Files processed: {result['files_processed']}")
print(f"Summary: {result['summary']}")
```

### Scan with Date Range

```python
from datetime import datetime, timedelta

# Scan files modified in the last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

task_ticket = {
    "task_id": "task_002",
    "workspace_path": "/path/to/workspace",
    "start_date": start_date,
    "end_date": end_date,
    "scan_all": False,
    "reindex": True
}

result = asyncio.run(agent.handle_task(task_ticket))
```

### Semantic Search

```python
# Search for relevant files
results = asyncio.run(
    agent.search_workspace("machine learning code", limit=5)
)

for result in results:
    print(f"File: {result['file_path']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Preview: {result['content_preview'][:100]}...\n")
```

### Using Individual Components

#### FileScanner

```python
from src.file_scanner import FileScanner
from datetime import datetime, timedelta

scanner = FileScanner("/path/to/workspace")

# Get all files
all_files = scanner.get_all_files()

# Scan changes in date range
end_date = datetime.now()
start_date = end_date - timedelta(days=7)
changed_files = scanner.scan_changes(start_date, end_date)
```

#### ContentParser

```python
from src.content_parser import ContentParser

parser = ContentParser()

# Parse a Python file
result = parser.parse_file("/path/to/script.py", "code")
print(result['summary'])
print(result['details']['functions'])
print(result['details']['classes'])
```

#### RAGManager

```python
from src.rag_manager import RAGManager

rag = RAGManager(storage_path="./data/qdrant_storage")

# Index a file
rag.index_file(
    file_path="/path/to/file.py",
    content="def hello(): print('world')",
    metadata={"file_type": "code", "size": 1024}
)

# Search
results = rag.search_similar("greeting function", limit=5)
```

## Streamlit Demo

Run the interactive demo application:

```bash
streamlit run demos/local_agent_demo.py
```

The demo provides:
- Interactive workspace scanning
- Real-time file analysis
- Semantic search interface
- RAG system monitoring

## Demo Test Scripts

Two educational test scripts demonstrate the Local Agent in action:

### End-to-End Pipeline Test

Simulates real-world task processing with different scenarios:

```bash
python tests/test_pipeline.py
```

Features:
- 3 test scenarios (weekly update, recent changes, full scan)
- Detailed input/output message formatting
- Processing statistics and summaries
- Shows complete task lifecycle

### Message Bus Simulation

Demonstrates integration with SignalMesh's message bus architecture:

```bash
python tests/test_message_bus_simulation.py
```

Features:
- Mock message bus with pub/sub pattern
- LocalAgent with message bus integration
- Mock Reply Agent that drafts emails
- Complete message flow visualization
- Shows inter-agent communication

Both scripts provide verbose, educational output with emojis and formatting to help understand how the Local Agent integrates into the larger SignalMesh system.

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_local_agent.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
signalmesh-local-agent/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── agent.py              # LocalAgent orchestrator
│   ├── file_scanner.py       # FileScanner component
│   ├── content_parser.py     # ContentParser component
│   └── rag_manager.py        # RAGManager component
├── data/
│   ├── simulated_workspace/  # Sample workspace for testing
│   └── qdrant_storage/       # Vector database storage (generated)
├── demos/
│   └── local_agent_demo.py   # Streamlit demo application
├── tests/
│   ├── test_local_agent.py           # Unit tests
│   ├── test_pipeline.py              # End-to-end pipeline demo
│   └── test_message_bus_simulation.py # Message bus integration demo
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore patterns
```

## Components

### FileScanner

Scans workspace directories and identifies files based on modification dates.

**Features:**
- Recursive directory scanning
- Date range filtering
- File type classification (.py, .csv, .ipynb, .md, etc.)
- Ignores common artifacts (__pycache__, .git, etc.)

### ContentParser

Extracts structured information from different file types.

**Supported Formats:**
- **Python (.py)**: Functions, classes, imports via AST parsing
- **CSV (.csv)**: Row/column counts, data types via pandas
- **Jupyter Notebooks (.ipynb)**: Cell counts and types via nbformat
- **Markdown (.md)**: Word counts, line counts, headers

### RAGManager

Provides vector-based semantic search using Qdrant and sentence transformers.

**Features:**
- Local vector storage (no Docker required)
- all-MiniLM-L6-v2 embeddings (384 dimensions)
- Cosine similarity search
- Batch indexing support
- Collection management

### LocalAgent

Main orchestrator that coordinates all operations.

**Capabilities:**
- Asynchronous task processing
- Workspace scanning and indexing
- Progress tracking and reporting
- Error handling and logging

## Configuration

### Logging

The project uses loguru for logging. Configure logging in your application:

```python
from loguru import logger

# Configure logging level
logger.remove()  # Remove default handler
logger.add("file.log", level="DEBUG")  # Log to file
logger.add(sys.stderr, level="INFO")   # Log to console
```

### RAG Storage

By default, vector data is stored in `./data/qdrant_storage/`. Customize:

```python
agent = LocalAgent(
    workspace_path="/path/to/workspace",
    rag_storage_path="/custom/qdrant/path"
)
```

## Dependencies

Core dependencies:
- **qdrant-client**: Vector database client
- **sentence-transformers**: Text embedding models
- **pandas**: Data analysis for CSV parsing
- **nbformat**: Jupyter notebook parsing
- **pydantic**: Data validation
- **loguru**: Logging
- **streamlit**: Demo application

See `requirements.txt` for complete list.

## Troubleshooting

### Qdrant Version Compatibility Error

If you encounter a Pydantic validation error when initializing RAGManager:

```
pydantic_core._pydantic_core.ValidationError: 10 validation errors for CreateCollection
```

**Solution 1: Clear RAG Storage** (Recommended)

```bash
# Delete the old storage directory
rm -rf data/qdrant_storage

# Or use the helper script
python scripts/clear_rag_storage.py
```

**Solution 2: Update Dependencies**

```bash
# Reinstall with updated versions
pip install --upgrade qdrant-client pydantic
```

The storage will be automatically recreated with the correct schema when you next run the agent.

### Missing Dependencies

If you get `ModuleNotFoundError`:

```bash
# Make sure you're in the project directory
cd signalmesh-local-agent

# Install all dependencies
pip install -r requirements.txt
```

### Permission Errors

If you get permission errors when scanning workspace:

- Ensure the workspace path exists and is readable
- Check file permissions: `ls -la /path/to/workspace`
- Try using an absolute path instead of relative path

## Development

### Code Quality

The project follows Python best practices:
- Type hints on all functions
- Comprehensive docstrings
- Pydantic models for data validation
- Error handling with logging
- Unit tests with pytest

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## Roadmap

- [ ] Support for additional file types (JSON, YAML, etc.)
- [ ] LLM integration for intelligent summarization
- [ ] Real-time file monitoring with watchdog
- [ ] Multi-language support for code parsing
- [ ] Advanced filtering and query capabilities
- [ ] Export and reporting features
- [ ] Integration with other SignalMesh components

## License

[Add your license here]

## Contact

[Add contact information]

## Acknowledgments

Built with:
- [Qdrant](https://qdrant.tech/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [Streamlit](https://streamlit.io/) - Demo interface
