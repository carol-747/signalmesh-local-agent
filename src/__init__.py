"""
SignalMesh Local Agent - Multi-agent research operations system.

This package provides the core functionality for the Local Agent component,
including file scanning, content parsing, and RAG-based indexing.
"""

__version__ = "0.1.0"

from .agent import LocalAgent
from .file_scanner import FileScanner
from .content_parser import ContentParser
from .rag_manager import RAGManager

__all__ = ["LocalAgent", "FileScanner", "ContentParser", "RAGManager"]
