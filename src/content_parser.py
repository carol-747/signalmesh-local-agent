"""
Content Parser Module for SignalMesh Local Agent.

This module provides functionality to parse different file types and extract
structured information from them.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


class ParseResult(BaseModel):
    """Result of parsing a file."""

    file_path: str = Field(..., description="Path to the parsed file")
    file_type: str = Field(..., description="Type of file")
    summary: str = Field(..., description="Human-readable summary")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details about the file"
    )
    content_preview: str = Field(
        default="",
        description="Preview of file content"
    )
    success: bool = Field(default=True, description="Whether parsing succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ContentParser:
    """
    Parser for extracting structured information from various file types.

    Supports parsing of Python files, CSV data files, Jupyter notebooks,
    and Markdown documents.
    """

    def __init__(self, max_preview_length: int = 500) -> None:
        """
        Initialize the ContentParser.

        Args:
            max_preview_length: Maximum length of content preview in characters
        """
        self.max_preview_length = max_preview_length
        logger.info("ContentParser initialized")

    def parse_file(self, file_path: str, file_type: str) -> Dict:
        """
        Parse a file and extract structured information.

        Args:
            file_path: Path to the file to parse
            file_type: Type of file (code, data, notebook, note, other)

        Returns:
            Dictionary containing parse results

        Example:
            >>> parser = ContentParser()
            >>> result = parser.parse_file("/path/to/script.py", "code")
            >>> print(result['summary'])
        """
        path = Path(file_path)

        if not path.exists():
            return ParseResult(
                file_path=file_path,
                file_type=file_type,
                summary="File not found",
                success=False,
                error="File does not exist"
            ).model_dump()

        logger.debug(f"Parsing file: {file_path} (type: {file_type})")

        try:
            if file_type == "code" and path.suffix == ".py":
                return self._parse_python(path)
            elif file_type == "data" and path.suffix == ".csv":
                return self._parse_csv(path)
            elif file_type == "notebook" and path.suffix == ".ipynb":
                return self._parse_notebook(path)
            elif file_type == "note" and path.suffix == ".md":
                return self._parse_markdown(path)
            else:
                return self._parse_generic(path, file_type)

        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                file_type=file_type,
                summary=f"Failed to parse file: {str(e)}",
                success=False,
                error=str(e)
            ).model_dump()

    def _parse_python(self, path: Path) -> Dict:
        """
        Parse a Python file and extract functions and classes.

        Args:
            path: Path to the Python file

        Returns:
            ParseResult dictionary
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            functions: List[str] = []
            classes: List[str] = []
            imports: List[str] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)

            # Generate summary
            parts = []
            if classes:
                parts.append(f"{len(classes)} class(es)")
            if functions:
                parts.append(f"{len(functions)} function(s)")

            summary = f"Python file with {', '.join(parts) if parts else 'no classes or functions'}"

            # Content preview
            lines = content.split('\n')
            preview_lines = lines[:20]  # First 20 lines
            preview = '\n'.join(preview_lines)
            if len(preview) > self.max_preview_length:
                preview = preview[:self.max_preview_length] + "..."

            return ParseResult(
                file_path=str(path),
                file_type="code",
                summary=summary,
                details={
                    "classes": classes,
                    "functions": functions,
                    "imports": imports[:10],  # Limit imports
                    "lines_of_code": len(lines),
                },
                content_preview=preview
            ).model_dump()

        except SyntaxError as e:
            logger.warning(f"Syntax error in Python file {path}: {e}")
            return ParseResult(
                file_path=str(path),
                file_type="code",
                summary="Python file with syntax errors",
                details={"error": str(e)},
                success=False,
                error=f"Syntax error: {str(e)}"
            ).model_dump()

    def _parse_csv(self, path: Path) -> Dict:
        """
        Parse a CSV file and extract metadata.

        Args:
            path: Path to the CSV file

        Returns:
            ParseResult dictionary
        """
        try:
            # Read CSV with pandas
            df = pd.read_csv(path)

            rows, cols = df.shape
            columns = df.columns.tolist()
            dtypes = df.dtypes.astype(str).to_dict()

            summary = f"CSV file with {rows} rows and {cols} columns"

            # Create preview (first few rows)
            preview = df.head(5).to_string()
            if len(preview) > self.max_preview_length:
                preview = preview[:self.max_preview_length] + "..."

            return ParseResult(
                file_path=str(path),
                file_type="data",
                summary=summary,
                details={
                    "rows": rows,
                    "columns": cols,
                    "column_names": columns,
                    "dtypes": dtypes,
                },
                content_preview=preview
            ).model_dump()

        except pd.errors.EmptyDataError:
            return ParseResult(
                file_path=str(path),
                file_type="data",
                summary="Empty CSV file",
                details={"rows": 0, "columns": 0},
            ).model_dump()

        except Exception as e:
            logger.error(f"Error parsing CSV {path}: {e}")
            return ParseResult(
                file_path=str(path),
                file_type="data",
                summary=f"CSV file (parse error: {str(e)})",
                success=False,
                error=str(e)
            ).model_dump()

    def _parse_notebook(self, path: Path) -> Dict:
        """
        Parse a Jupyter notebook and extract cell information.

        Args:
            path: Path to the notebook file

        Returns:
            ParseResult dictionary
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            code_cells = 0
            markdown_cells = 0
            total_cells = len(notebook.cells)

            for cell in notebook.cells:
                if cell.cell_type == "code":
                    code_cells += 1
                elif cell.cell_type == "markdown":
                    markdown_cells += 1

            summary = (
                f"Jupyter notebook with {total_cells} cells "
                f"({code_cells} code, {markdown_cells} markdown)"
            )

            # Create preview from first few cells
            preview_parts = []
            for i, cell in enumerate(notebook.cells[:3]):
                cell_type = cell.cell_type
                source = cell.source[:200] if cell.source else ""
                preview_parts.append(f"[Cell {i+1} - {cell_type}]\n{source}")

            preview = "\n\n".join(preview_parts)
            if len(preview) > self.max_preview_length:
                preview = preview[:self.max_preview_length] + "..."

            return ParseResult(
                file_path=str(path),
                file_type="notebook",
                summary=summary,
                details={
                    "total_cells": total_cells,
                    "code_cells": code_cells,
                    "markdown_cells": markdown_cells,
                },
                content_preview=preview
            ).model_dump()

        except Exception as e:
            logger.error(f"Error parsing notebook {path}: {e}")
            return ParseResult(
                file_path=str(path),
                file_type="notebook",
                summary=f"Jupyter notebook (parse error)",
                success=False,
                error=str(e)
            ).model_dump()

    def _parse_markdown(self, path: Path) -> Dict:
        """
        Parse a Markdown file and extract basic statistics.

        Args:
            path: Path to the Markdown file

        Returns:
            ParseResult dictionary
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split('\n')
            words = content.split()

            # Count headers
            headers = [line for line in lines if line.strip().startswith('#')]

            summary = (
                f"Markdown file with {len(words)} words, "
                f"{len(lines)} lines, {len(headers)} headers"
            )

            # Preview
            preview = content[:self.max_preview_length]
            if len(content) > self.max_preview_length:
                preview += "..."

            return ParseResult(
                file_path=str(path),
                file_type="note",
                summary=summary,
                details={
                    "words": len(words),
                    "lines": len(lines),
                    "headers": len(headers),
                },
                content_preview=preview
            ).model_dump()

        except Exception as e:
            logger.error(f"Error parsing markdown {path}: {e}")
            return ParseResult(
                file_path=str(path),
                file_type="note",
                summary=f"Markdown file (parse error)",
                success=False,
                error=str(e)
            ).model_dump()

    def _parse_generic(self, path: Path, file_type: str) -> Dict:
        """
        Parse a generic file and extract basic information.

        Args:
            path: Path to the file
            file_type: Type classification

        Returns:
            ParseResult dictionary
        """
        try:
            # Try to read as text
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                lines = content.split('\n')
                words = content.split()

                summary = f"{file_type.capitalize()} file with {len(lines)} lines"

                preview = content[:self.max_preview_length]
                if len(content) > self.max_preview_length:
                    preview += "..."

                return ParseResult(
                    file_path=str(path),
                    file_type=file_type,
                    summary=summary,
                    details={
                        "lines": len(lines),
                        "words": len(words),
                        "size_bytes": path.stat().st_size,
                    },
                    content_preview=preview
                ).model_dump()

            except UnicodeDecodeError:
                # Binary file
                size = path.stat().st_size
                return ParseResult(
                    file_path=str(path),
                    file_type=file_type,
                    summary=f"Binary {file_type} file ({size} bytes)",
                    details={"size_bytes": size},
                    content_preview="[Binary content]"
                ).model_dump()

        except Exception as e:
            logger.error(f"Error parsing generic file {path}: {e}")
            return ParseResult(
                file_path=str(path),
                file_type=file_type,
                summary=f"{file_type.capitalize()} file (parse error)",
                success=False,
                error=str(e)
            ).model_dump()
