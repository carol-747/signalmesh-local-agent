"""
Unit tests for SignalMesh Local Agent components.

This module contains tests for FileScanner, ContentParser, RAGManager,
and LocalAgent classes.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.agent import LocalAgent
from src.content_parser import ContentParser
from src.file_scanner import FileScanner
from src.rag_manager import RAGManager


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create sample Python file
        python_file = workspace / "sample.py"
        python_file.write_text("""
def hello_world():
    print("Hello, world!")

class SampleClass:
    def __init__(self):
        self.value = 42
""")

        # Create sample CSV file
        csv_file = workspace / "data.csv"
        csv_file.write_text("""name,age,city
Alice,30,New York
Bob,25,San Francisco
Charlie,35,Chicago
""")

        # Create sample Markdown file
        md_file = workspace / "notes.md"
        md_file.write_text("""# Sample Notes

## Introduction

This is a sample markdown file for testing.

## Features

- Feature 1
- Feature 2
- Feature 3
""")

        # Create a subdirectory with files
        subdir = workspace / "subdir"
        subdir.mkdir()

        nested_file = subdir / "nested.py"
        nested_file.write_text("""
import numpy as np

def calculate(x, y):
    return x + y
""")

        # Create __pycache__ to test ignoring
        pycache = workspace / "__pycache__"
        pycache.mkdir()
        (pycache / "sample.pyc").write_text("compiled")

        yield workspace


@pytest.fixture
def temp_rag_storage():
    """Create a temporary directory for RAG storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestFileScanner:
    """Tests for FileScanner class."""

    def test_initialization(self, temp_workspace):
        """Test FileScanner initialization."""
        scanner = FileScanner(str(temp_workspace))
        assert scanner.workspace_path == temp_workspace

    def test_initialization_invalid_path(self):
        """Test FileScanner with invalid path."""
        with pytest.raises(ValueError):
            FileScanner("/nonexistent/path")

    def test_get_all_files(self, temp_workspace):
        """Test scanning all files."""
        scanner = FileScanner(str(temp_workspace))
        files = scanner.get_all_files()

        # Should find .py, .csv, .md files (not .pyc)
        assert len(files) >= 4

        # Check that __pycache__ files are ignored
        file_paths = [f["path"] for f in files]
        assert not any("__pycache__" in path for path in file_paths)

        # Check file metadata structure
        for file_meta in files:
            assert "path" in file_meta
            assert "name" in file_meta
            assert "file_type" in file_meta
            assert "modified" in file_meta
            assert "size" in file_meta

    def test_scan_changes_date_range(self, temp_workspace):
        """Test scanning with date range."""
        scanner = FileScanner(str(temp_workspace))

        # Scan last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        files = scanner.scan_changes(start_date, end_date)

        # Should find all recently created files
        assert len(files) >= 4

    def test_scan_changes_future_date(self, temp_workspace):
        """Test scanning with future date range (should find nothing)."""
        scanner = FileScanner(str(temp_workspace))

        # Scan future dates
        start_date = datetime.now() + timedelta(days=1)
        end_date = datetime.now() + timedelta(days=7)

        files = scanner.scan_changes(start_date, end_date)

        # Should find no files
        assert len(files) == 0

    def test_file_type_classification(self, temp_workspace):
        """Test file type classification."""
        scanner = FileScanner(str(temp_workspace))
        files = scanner.get_all_files()

        file_types = {f["name"]: f["file_type"] for f in files}

        assert file_types.get("sample.py") == "code"
        assert file_types.get("data.csv") == "data"
        assert file_types.get("notes.md") == "note"


class TestContentParser:
    """Tests for ContentParser class."""

    def test_initialization(self):
        """Test ContentParser initialization."""
        parser = ContentParser()
        assert parser.max_preview_length == 500

    def test_parse_python_file(self, temp_workspace):
        """Test parsing Python files."""
        parser = ContentParser()
        python_file = temp_workspace / "sample.py"

        result = parser.parse_file(str(python_file), "code")

        assert result["success"] is True
        assert result["file_type"] == "code"
        assert "hello_world" in result["details"]["functions"]
        assert "SampleClass" in result["details"]["classes"]

    def test_parse_csv_file(self, temp_workspace):
        """Test parsing CSV files."""
        parser = ContentParser()
        csv_file = temp_workspace / "data.csv"

        result = parser.parse_file(str(csv_file), "data")

        assert result["success"] is True
        assert result["file_type"] == "data"
        assert result["details"]["rows"] == 3
        assert result["details"]["columns"] == 3
        assert "name" in result["details"]["column_names"]

    def test_parse_markdown_file(self, temp_workspace):
        """Test parsing Markdown files."""
        parser = ContentParser()
        md_file = temp_workspace / "notes.md"

        result = parser.parse_file(str(md_file), "note")

        assert result["success"] is True
        assert result["file_type"] == "note"
        assert result["details"]["headers"] >= 2
        assert result["details"]["words"] > 0

    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        parser = ContentParser()
        result = parser.parse_file("/nonexistent/file.py", "code")

        assert result["success"] is False
        assert "error" in result

    def test_parse_python_syntax_error(self, temp_workspace):
        """Test parsing Python file with syntax error."""
        parser = ContentParser()
        bad_python = temp_workspace / "bad.py"
        bad_python.write_text("def incomplete_function(")

        result = parser.parse_file(str(bad_python), "code")

        assert result["success"] is False
        assert "error" in result


class TestRAGManager:
    """Tests for RAGManager class."""

    def test_initialization(self, temp_rag_storage):
        """Test RAGManager initialization."""
        rag = RAGManager(storage_path=temp_rag_storage)

        assert rag.collection_name == "local_rag"
        assert rag.embedding_dim == 384  # all-MiniLM-L6-v2 dimension

    def test_index_file(self, temp_rag_storage):
        """Test indexing a single file."""
        rag = RAGManager(storage_path=temp_rag_storage)

        point_id = rag.index_file(
            file_path="/test/file.py",
            content="This is a test file with some content",
            metadata={"file_type": "code"}
        )

        assert point_id is not None

        # Check collection info
        info = rag.get_collection_info()
        assert info["points_count"] == 1

    def test_index_multiple_files(self, temp_rag_storage):
        """Test indexing multiple files."""
        rag = RAGManager(storage_path=temp_rag_storage)

        files = [
            {
                "file_path": "/test/file1.py",
                "content": "Machine learning code for classification",
                "metadata": {"file_type": "code"}
            },
            {
                "file_path": "/test/file2.py",
                "content": "Data processing and analysis",
                "metadata": {"file_type": "code"}
            }
        ]

        point_ids = rag.index_multiple_files(files)

        assert len(point_ids) == 2

        info = rag.get_collection_info()
        assert info["points_count"] == 2

    def test_search_similar(self, temp_rag_storage):
        """Test semantic search."""
        rag = RAGManager(storage_path=temp_rag_storage)

        # Index some files
        files = [
            {
                "file_path": "/test/ml_code.py",
                "content": "Machine learning model training and evaluation",
                "metadata": {"file_type": "code"}
            },
            {
                "file_path": "/test/data_prep.py",
                "content": "Data preprocessing and cleaning functions",
                "metadata": {"file_type": "code"}
            },
            {
                "file_path": "/test/viz.py",
                "content": "Visualization and plotting utilities",
                "metadata": {"file_type": "code"}
            }
        ]

        rag.index_multiple_files(files)

        # Search for ML-related content
        results = rag.search_similar("machine learning", limit=2)

        assert len(results) <= 2
        assert results[0]["file_path"] == "/test/ml_code.py"
        assert results[0]["score"] > 0.5  # Should have good similarity

    def test_get_collection_info(self, temp_rag_storage):
        """Test getting collection information."""
        rag = RAGManager(storage_path=temp_rag_storage)

        info = rag.get_collection_info()

        assert "collection_name" in info
        assert "points_count" in info
        assert "status" in info

    def test_clear_collection(self, temp_rag_storage):
        """Test clearing collection."""
        rag = RAGManager(storage_path=temp_rag_storage)

        # Index a file
        rag.index_file(
            file_path="/test/file.py",
            content="Test content",
            metadata={}
        )

        # Clear collection
        rag.clear_collection()

        # Should be empty
        info = rag.get_collection_info()
        assert info["points_count"] == 0


class TestLocalAgent:
    """Tests for LocalAgent class."""

    def test_initialization(self, temp_workspace, temp_rag_storage):
        """Test LocalAgent initialization."""
        agent = LocalAgent(
            workspace_path=str(temp_workspace),
            rag_storage_path=temp_rag_storage
        )

        assert agent.workspace_path == temp_workspace
        assert agent.file_scanner is not None
        assert agent.content_parser is not None
        assert agent.rag_manager is not None

    def test_initialization_invalid_workspace(self):
        """Test LocalAgent with invalid workspace."""
        with pytest.raises(ValueError):
            LocalAgent("/nonexistent/workspace")

    @pytest.mark.asyncio
    async def test_handle_task_scan_all(self, temp_workspace, temp_rag_storage):
        """Test handling a task to scan all files."""
        agent = LocalAgent(
            workspace_path=str(temp_workspace),
            rag_storage_path=temp_rag_storage
        )

        task_ticket = {
            "task_id": "test_task_001",
            "workspace_path": str(temp_workspace),
            "scan_all": True,
            "reindex": True
        }

        result = await agent.handle_task(task_ticket)

        assert result["status"] == "success"
        assert result["files_processed"] >= 4
        assert result["files_indexed"] >= 4
        assert len(result["file_list"]) >= 4
        assert result["summary"] != ""

    @pytest.mark.asyncio
    async def test_handle_task_date_range(self, temp_workspace, temp_rag_storage):
        """Test handling a task with date range."""
        agent = LocalAgent(
            workspace_path=str(temp_workspace),
            rag_storage_path=temp_rag_storage
        )

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        task_ticket = {
            "task_id": "test_task_002",
            "workspace_path": str(temp_workspace),
            "start_date": start_date,
            "end_date": end_date,
            "scan_all": False,
            "reindex": True
        }

        result = await agent.handle_task(task_ticket)

        assert result["status"] == "success"
        assert result["files_processed"] >= 4

    @pytest.mark.asyncio
    async def test_search_workspace(self, temp_workspace, temp_rag_storage):
        """Test workspace search."""
        agent = LocalAgent(
            workspace_path=str(temp_workspace),
            rag_storage_path=temp_rag_storage
        )

        # First, scan and index the workspace
        task_ticket = {
            "task_id": "test_task_003",
            "workspace_path": str(temp_workspace),
            "scan_all": True,
            "reindex": True
        }

        await agent.handle_task(task_ticket)

        # Now search
        results = await agent.search_workspace("python code", limit=3)

        assert len(results) <= 3
        assert len(results) > 0

    def test_get_rag_status(self, temp_workspace, temp_rag_storage):
        """Test getting RAG status."""
        agent = LocalAgent(
            workspace_path=str(temp_workspace),
            rag_storage_path=temp_rag_storage
        )

        status = agent.get_rag_status()

        assert "collection_name" in status
        assert "points_count" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
