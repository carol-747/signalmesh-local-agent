"""
Local Agent Module for SignalMesh.

This module provides the main orchestrator for the Local Agent component,
coordinating file scanning, parsing, and RAG indexing operations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from .content_parser import ContentParser
from .file_scanner import FileScanner
from .rag_manager import RAGManager


class TaskTicket(BaseModel):
    """Task ticket for agent processing."""

    task_id: str = Field(..., description="Unique task identifier")
    workspace_path: str = Field(..., description="Path to workspace")
    start_date: Optional[datetime] = Field(
        None,
        description="Start date for file scanning"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="End date for file scanning"
    )
    scan_all: bool = Field(
        default=False,
        description="Scan all files regardless of date"
    )
    reindex: bool = Field(
        default=True,
        description="Whether to reindex files in RAG"
    )


class TaskResult(BaseModel):
    """Result of task processing."""

    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Status (success/error)")
    message: str = Field(..., description="Status message")
    files_processed: int = Field(default=0, description="Number of files processed")
    files_indexed: int = Field(default=0, description="Number of files indexed")
    summary: str = Field(default="", description="Generated summary")
    file_list: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of processed files"
    )
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")


class LocalAgent:
    """
    Main orchestrator for the Local Agent component.

    The LocalAgent coordinates file scanning, content parsing, and RAG indexing
    to provide semantic search capabilities over workspace files.
    """

    def __init__(
        self,
        workspace_path: str,
        rag_storage_path: str = "./data/qdrant_storage"
    ) -> None:
        """
        Initialize the Local Agent.

        Args:
            workspace_path: Path to the workspace directory
            rag_storage_path: Path for RAG vector storage

        Raises:
            ValueError: If workspace_path is invalid

        Example:
            >>> agent = LocalAgent("/path/to/workspace")
            >>> import asyncio
            >>> result = asyncio.run(agent.handle_task({
            ...     "task_id": "task_001",
            ...     "workspace_path": "/path/to/workspace",
            ...     "scan_all": True
            ... }))
        """
        self.workspace_path = Path(workspace_path).resolve()

        if not self.workspace_path.exists():
            raise ValueError(f"Workspace path does not exist: {workspace_path}")

        logger.info(f"Initializing LocalAgent for workspace: {self.workspace_path}")

        # Initialize components
        self.file_scanner = FileScanner(str(self.workspace_path))
        self.content_parser = ContentParser()
        self.rag_manager = RAGManager(storage_path=rag_storage_path)

        logger.info("LocalAgent initialized successfully")

    async def handle_task(self, task_ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a task ticket asynchronously.

        This is the main entry point for processing tasks. It coordinates
        scanning, parsing, and indexing operations.

        Args:
            task_ticket: Dictionary with task parameters

        Returns:
            Dictionary containing task results

        Example:
            >>> import asyncio
            >>> task = {
            ...     "task_id": "scan_001",
            ...     "workspace_path": "/path/to/workspace",
            ...     "start_date": datetime(2024, 1, 1),
            ...     "end_date": datetime(2024, 12, 31),
            ...     "scan_all": False,
            ...     "reindex": True
            ... }
            >>> result = asyncio.run(agent.handle_task(task))
        """
        start_time = datetime.now()

        try:
            # Validate and parse task ticket
            ticket = TaskTicket(**task_ticket)
            logger.info(f"Processing task: {ticket.task_id}")

            # Step 1: Scan workspace for files
            logger.info("Step 1: Scanning workspace...")
            if ticket.scan_all:
                changed_files = self.file_scanner.get_all_files()
            else:
                if not ticket.start_date or not ticket.end_date:
                    raise ValueError(
                        "start_date and end_date required when scan_all=False"
                    )
                changed_files = self.file_scanner.scan_changes(
                    ticket.start_date,
                    ticket.end_date
                )

            if not changed_files:
                logger.info("No files found matching criteria")
                return TaskResult(
                    task_id=ticket.task_id,
                    status="success",
                    message="No files found matching criteria",
                    files_processed=0,
                    files_indexed=0,
                    processing_time=(datetime.now() - start_time).total_seconds()
                ).model_dump()

            logger.info(f"Found {len(changed_files)} files")

            # Step 2: Parse files
            logger.info("Step 2: Parsing file contents...")
            parsed_files = []
            parse_errors = []

            for file_meta in changed_files:
                try:
                    parse_result = self.content_parser.parse_file(
                        file_meta["path"],
                        file_meta["file_type"]
                    )
                    parsed_files.append({
                        **file_meta,
                        "parse_result": parse_result
                    })

                    if not parse_result.get("success", True):
                        parse_errors.append(
                            f"{file_meta['name']}: {parse_result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    logger.error(f"Error parsing {file_meta['path']}: {e}")
                    parse_errors.append(f"{file_meta['name']}: {str(e)}")

            logger.info(f"Successfully parsed {len(parsed_files)} files")

            # Step 3: Index in RAG (if requested)
            files_indexed = 0
            if ticket.reindex:
                logger.info("Step 3: Indexing files in RAG...")
                files_indexed = await self._index_files(parsed_files)
                logger.info(f"Indexed {files_indexed} files in RAG")

            # Step 4: Generate summary
            logger.info("Step 4: Generating summary...")
            summary = self._generate_summary(parsed_files)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Build result
            result = TaskResult(
                task_id=ticket.task_id,
                status="success",
                message=f"Successfully processed {len(changed_files)} files",
                files_processed=len(changed_files),
                files_indexed=files_indexed,
                summary=summary,
                file_list=changed_files,
                errors=parse_errors,
                processing_time=processing_time
            )

            logger.info(
                f"Task {ticket.task_id} completed in {processing_time:.2f}s"
            )

            return result.model_dump()

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return TaskResult(
                task_id=task_ticket.get("task_id", "unknown"),
                status="error",
                message=f"Task failed: {str(e)}",
                errors=[str(e)],
                processing_time=processing_time
            ).model_dump()

    async def _index_files(self, parsed_files: List[Dict[str, Any]]) -> int:
        """
        Index parsed files in the RAG system.

        Args:
            parsed_files: List of parsed file dictionaries

        Returns:
            Number of files successfully indexed
        """
        indexed_count = 0

        for file_data in parsed_files:
            try:
                parse_result = file_data.get("parse_result", {})

                # Skip files that failed to parse
                if not parse_result.get("success", True):
                    continue

                # Prepare content for indexing
                content_parts = [
                    parse_result.get("summary", ""),
                    parse_result.get("content_preview", "")
                ]

                # Add details as text
                details = parse_result.get("details", {})
                if details:
                    for key, value in details.items():
                        if isinstance(value, (list, dict)):
                            content_parts.append(f"{key}: {str(value)[:200]}")
                        else:
                            content_parts.append(f"{key}: {value}")

                content = "\n".join(filter(None, content_parts))

                # Prepare metadata
                metadata = {
                    "file_name": file_data.get("name", ""),
                    "file_type": file_data.get("file_type", ""),
                    "size": file_data.get("size", 0),
                    "modified": file_data.get("modified", datetime.now()).isoformat(),
                    "summary": parse_result.get("summary", "")
                }

                # Index in RAG
                self.rag_manager.index_file(
                    file_path=file_data.get("path", ""),
                    content=content,
                    metadata=metadata
                )

                indexed_count += 1

            except Exception as e:
                logger.error(
                    f"Error indexing file {file_data.get('path', 'unknown')}: {e}"
                )

        return indexed_count

    def _generate_summary(self, parsed_files: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of processed files.

        For now, this uses deterministic summarization. In production,
        this could call an LLM API for more sophisticated summaries.

        Args:
            parsed_files: List of parsed file dictionaries

        Returns:
            Summary text
        """
        if not parsed_files:
            return "No files were processed."

        # Count by file type
        type_counts: Dict[str, int] = {}
        for file_data in parsed_files:
            file_type = file_data.get("file_type", "other")
            type_counts[file_type] = type_counts.get(file_type, 0) + 1

        # Build summary parts
        summary_parts = [
            f"Processed {len(parsed_files)} files from workspace.",
            "",
            "File breakdown:"
        ]

        for file_type, count in sorted(type_counts.items()):
            summary_parts.append(f"  - {file_type}: {count} file(s)")

        # Add some statistics
        total_size = sum(f.get("size", 0) for f in parsed_files)
        size_mb = total_size / (1024 * 1024)

        summary_parts.extend([
            "",
            f"Total size: {size_mb:.2f} MB",
        ])

        # List some notable files
        code_files = [
            f for f in parsed_files
            if f.get("file_type") == "code"
        ]

        if code_files:
            summary_parts.extend([
                "",
                "Notable code files:"
            ])
            for file_data in code_files[:5]:  # Top 5
                parse_result = file_data.get("parse_result", {})
                file_summary = parse_result.get("summary", "")
                summary_parts.append(
                    f"  - {file_data.get('name', 'unknown')}: {file_summary}"
                )

        return "\n".join(summary_parts)

    async def search_workspace(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the workspace using semantic search.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results

        Example:
            >>> import asyncio
            >>> results = asyncio.run(
            ...     agent.search_workspace("machine learning code", limit=5)
            ... )
        """
        logger.info(f"Searching workspace for: {query}")

        try:
            results = self.rag_manager.search_similar(query, limit=limit)
            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def get_rag_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system.

        Returns:
            Dictionary with RAG statistics

        Example:
            >>> status = agent.get_rag_status()
            >>> print(f"Files indexed: {status['points_count']}")
        """
        try:
            return self.rag_manager.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting RAG status: {e}")
            return {"error": str(e)}
