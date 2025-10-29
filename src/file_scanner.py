"""
File Scanner Module for SignalMesh Local Agent.

This module provides functionality to scan workspace directories and identify
files modified within a specified date range.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from loguru import logger
from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    """Metadata for a scanned file."""

    path: str = Field(..., description="Absolute path to the file")
    name: str = Field(..., description="File name")
    file_type: str = Field(..., description="Classified file type")
    modified: datetime = Field(..., description="Last modified timestamp")
    size: int = Field(..., description="File size in bytes")


class FileScanner:
    """
    Scanner for workspace files with filtering by modification date.

    The FileScanner walks through a workspace directory and identifies files
    that have been modified within a specified date range. It classifies files
    by extension and ignores common development artifacts.
    """

    # File type mappings
    FILE_TYPE_MAP: Dict[str, str] = {
        ".py": "code",
        ".csv": "data",
        ".ipynb": "notebook",
        ".md": "note",
        ".txt": "note",
        ".json": "data",
        ".yaml": "data",
        ".yml": "data",
    }

    # Directories and patterns to ignore
    IGNORE_PATTERNS: List[str] = [
        ".git",
        "__pycache__",
        ".ipynb_checkpoints",
        "venv",
        ".venv",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        "*.pyc",
        ".DS_Store",
    ]

    def __init__(self, workspace_path: str) -> None:
        """
        Initialize the FileScanner.

        Args:
            workspace_path: Path to the workspace directory to scan

        Raises:
            ValueError: If workspace_path doesn't exist or isn't a directory
        """
        self.workspace_path = Path(workspace_path).resolve()

        if not self.workspace_path.exists():
            raise ValueError(f"Workspace path does not exist: {workspace_path}")

        if not self.workspace_path.is_dir():
            raise ValueError(f"Workspace path is not a directory: {workspace_path}")

        logger.info(f"FileScanner initialized for workspace: {self.workspace_path}")

    def _should_ignore(self, path: Path) -> bool:
        """
        Check if a path should be ignored based on ignore patterns.

        Args:
            path: Path to check

        Returns:
            True if the path should be ignored, False otherwise
        """
        path_str = str(path)
        path_parts = path.parts

        for pattern in self.IGNORE_PATTERNS:
            # Check if any part of the path matches ignore patterns
            if pattern in path_parts:
                return True

            # Check for file patterns
            if pattern.startswith("*") and path.name.endswith(pattern[1:]):
                return True

            # Check exact match
            if pattern in path_str:
                return True

        return False

    def _classify_file_type(self, file_path: Path) -> str:
        """
        Classify a file by its extension.

        Args:
            file_path: Path to the file

        Returns:
            Classified file type (code, data, notebook, note, or other)
        """
        extension = file_path.suffix.lower()
        return self.FILE_TYPE_MAP.get(extension, "other")

    def scan_changes(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Scan workspace for files modified within a date range.

        Args:
            start_date: Start of the date range (inclusive)
            end_date: End of the date range (inclusive)

        Returns:
            List of dictionaries containing file metadata for files modified
            within the specified date range

        Example:
            >>> scanner = FileScanner("/path/to/workspace")
            >>> from datetime import datetime, timedelta
            >>> end = datetime.now()
            >>> start = end - timedelta(days=7)
            >>> changed_files = scanner.scan_changes(start, end)
        """
        logger.info(
            f"Scanning workspace for changes between {start_date} and {end_date}"
        )

        changed_files: List[Dict] = []
        scanned_count = 0
        ignored_count = 0

        try:
            for root, dirs, files in os.walk(self.workspace_path):
                root_path = Path(root)

                # Modify dirs in-place to prevent descending into ignored directories
                dirs[:] = [
                    d for d in dirs
                    if not self._should_ignore(root_path / d)
                ]

                for file_name in files:
                    file_path = root_path / file_name
                    scanned_count += 1

                    # Skip ignored files
                    if self._should_ignore(file_path):
                        ignored_count += 1
                        continue

                    try:
                        # Get file modification time
                        stat_info = file_path.stat()
                        modified_time = datetime.fromtimestamp(stat_info.st_mtime)

                        # Check if file was modified within date range
                        if start_date <= modified_time <= end_date:
                            file_type = self._classify_file_type(file_path)

                            metadata = FileMetadata(
                                path=str(file_path),
                                name=file_name,
                                file_type=file_type,
                                modified=modified_time,
                                size=stat_info.st_size,
                            )

                            changed_files.append(metadata.model_dump())
                            logger.debug(
                                f"Found changed file: {file_name} "
                                f"(type: {file_type}, modified: {modified_time})"
                            )

                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not access file {file_path}: {e}")
                        continue

            logger.info(
                f"Scan complete: {len(changed_files)} changed files found, "
                f"{scanned_count} total files scanned, {ignored_count} ignored"
            )

            return changed_files

        except Exception as e:
            logger.error(f"Error during workspace scan: {e}")
            raise

    def get_all_files(self) -> List[Dict]:
        """
        Get all files in the workspace (no date filtering).

        Returns:
            List of dictionaries containing file metadata for all files
        """
        logger.info("Scanning all files in workspace")

        all_files: List[Dict] = []

        try:
            for root, dirs, files in os.walk(self.workspace_path):
                root_path = Path(root)

                # Modify dirs in-place to prevent descending into ignored directories
                dirs[:] = [
                    d for d in dirs
                    if not self._should_ignore(root_path / d)
                ]

                for file_name in files:
                    file_path = root_path / file_name

                    # Skip ignored files
                    if self._should_ignore(file_path):
                        continue

                    try:
                        stat_info = file_path.stat()
                        modified_time = datetime.fromtimestamp(stat_info.st_mtime)
                        file_type = self._classify_file_type(file_path)

                        metadata = FileMetadata(
                            path=str(file_path),
                            name=file_name,
                            file_type=file_type,
                            modified=modified_time,
                            size=stat_info.st_size,
                        )

                        all_files.append(metadata.model_dump())

                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not access file {file_path}: {e}")
                        continue

            logger.info(f"Found {len(all_files)} total files")
            return all_files

        except Exception as e:
            logger.error(f"Error during workspace scan: {e}")
            raise
