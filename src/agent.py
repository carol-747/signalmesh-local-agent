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
    """Result of task processing - LEGACY, kept for backwards compatibility."""

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
        rag_storage_path: str = "./data/qdrant_storage",
        agent_id: str = "local-agent-001",
        agent_version: str = "0.1.0"
    ) -> None:
        """
        Initialize the Local Agent.

        Args:
            workspace_path: Path to the workspace directory
            rag_storage_path: Path for RAG vector storage
            agent_id: Unique identifier for this agent instance
            agent_version: Version of the agent

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
        self.agent_id = agent_id
        self.agent_version = agent_version

        if not self.workspace_path.exists():
            raise ValueError(f"Workspace path does not exist: {workspace_path}")

        logger.info(f"Initializing LocalAgent for workspace: {self.workspace_path}")

        # Initialize components
        self.file_scanner = FileScanner(str(self.workspace_path))
        self.content_parser = ContentParser()
        self.rag_manager = RAGManager(storage_path=rag_storage_path)

        logger.info("LocalAgent initialized successfully")

    async def handle_task(
        self,
        task_ticket: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle a task ticket asynchronously with comprehensive output.

        This is the main entry point for processing tasks. It coordinates
        scanning, parsing, and indexing operations.

        Args:
            task_ticket: Dictionary with task parameters
            context: Optional context from original request (requester, question, etc.)

        Returns:
            Dictionary containing comprehensive task results

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
            >>> context = {
            ...     "requester": "Sarah Chen",
            ...     "requester_email": "sarah.chen@example.com",
            ...     "original_question": "What changed this week?"
            ... }
            >>> result = asyncio.run(agent.handle_task(task, context))
        """
        start_time = datetime.now()

        try:
            # Validate and parse task ticket
            ticket = TaskTicket(**task_ticket)
            logger.info(f"Processing task: {ticket.task_id}")

            # Extract context if passed through
            if context is None and "context" in task_ticket:
                context = task_ticket["context"]

            # Step 1: Scan workspace for files
            logger.info("Step 1: Scanning workspace...")
            scan_start = datetime.now()

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

            scan_time = (datetime.now() - scan_start).total_seconds()

            if not changed_files:
                logger.info("No files found matching criteria")
                return self._build_comprehensive_result(
                    task_ticket=task_ticket,
                    context=context,
                    start_time=start_time,
                    scan_time=scan_time,
                    parse_time=0,
                    index_time=0,
                    changed_files=[],
                    parsed_files=[],
                    files_indexed=0,
                    parse_errors=[],
                    message="No files found matching criteria"
                )

            logger.info(f"Found {len(changed_files)} files")

            # Step 2: Parse files
            logger.info("Step 2: Parsing file contents...")
            parse_start = datetime.now()
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

            parse_time = (datetime.now() - parse_start).total_seconds()
            logger.info(f"Successfully parsed {len(parsed_files)} files")

            # Step 3: Index in RAG (if requested)
            index_start = datetime.now()
            files_indexed = 0
            if ticket.reindex:
                logger.info("Step 3: Indexing files in RAG...")
                files_indexed = await self._index_files(parsed_files)
                logger.info(f"Indexed {files_indexed} files in RAG")
            index_time = (datetime.now() - index_start).total_seconds()

            # Step 4: Build comprehensive result
            logger.info("Step 4: Building comprehensive result...")
            result = self._build_comprehensive_result(
                task_ticket=task_ticket,
                context=context,
                start_time=start_time,
                scan_time=scan_time,
                parse_time=parse_time,
                index_time=index_time,
                changed_files=changed_files,
                parsed_files=parsed_files,
                files_indexed=files_indexed,
                parse_errors=parse_errors,
                message=f"Successfully processed {len(changed_files)} files"
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Task {ticket.task_id} completed in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return self._build_error_result(
                task_ticket=task_ticket,
                context=context,
                error=str(e),
                processing_time=processing_time
            )

    def _build_comprehensive_result(
        self,
        task_ticket: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        start_time: datetime,
        scan_time: float,
        parse_time: float,
        index_time: float,
        changed_files: List[Dict[str, Any]],
        parsed_files: List[Dict[str, Any]],
        files_indexed: int,
        parse_errors: List[str],
        message: str
    ) -> Dict[str, Any]:
        """Build comprehensive result with all fields."""

        processing_time = (datetime.now() - start_time).total_seconds()
        completed_at = datetime.now()

        # Generate statistics
        statistics = self._generate_statistics(parsed_files, files_indexed)

        # Generate breakdown by type
        breakdown = self._generate_breakdown_by_type(parsed_files)

        # Generate insights
        insights = self._generate_insights(parsed_files)

        # Generate enhanced file list
        enhanced_files = self._generate_enhanced_file_list(parsed_files)

        # Get RAG status
        rag_status = self._get_rag_status_comprehensive()

        # Generate summary
        summary = self._generate_summary(parsed_files)

        # Build comprehensive context
        comprehensive_context = self._build_context(task_ticket, context)

        # Build result
        result = {
            # Task identification
            "task_id": task_ticket.get("task_id", "unknown"),
            "agent_id": self.agent_id,
            "timestamp": completed_at.isoformat(),
            "status": "success",
            "message": message,

            # Original request context
            "context": comprehensive_context,

            # Processing results
            "results": {
                "summary": summary,
                "statistics": statistics,
                "breakdown_by_type": breakdown,
                "files": enhanced_files
            },

            # RAG system status
            "rag_status": rag_status,

            # Key insights
            "insights": insights,

            # Performance metrics
            "performance": {
                "processing_time_seconds": processing_time,
                "scan_time_seconds": scan_time,
                "parse_time_seconds": parse_time,
                "index_time_seconds": index_time,
                "started_at": start_time.isoformat(),
                "completed_at": completed_at.isoformat()
            },

            # Errors and warnings
            "errors": parse_errors,
            "warnings": self._generate_warnings(parsed_files),

            # Suggested actions
            "suggested_actions": self._generate_suggested_actions(parsed_files),

            # Metadata
            "metadata": {
                "agent_version": self.agent_version,
                "schema_version": "1.0",
                "reply_agent_compatible": True,
                "requires_user_action": False,
                "is_final": True
            }
        }

        return result

    def _build_context(
        self,
        task_ticket: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive context from task ticket and provided context."""

        if context is None:
            context = {}

        # Parse requester info
        requester_info = {
            "name": context.get("requester", "Unknown"),
            "email": context.get("requester_email", "unknown@example.com"),
            "role": context.get("requester_role", "User"),
            "timezone": context.get("timezone", "UTC")
        }

        # Parse request details
        request_info = {
            "original_question": context.get("original_question", ""),
            "channel": context.get("channel", "API"),
            "thread_id": context.get("thread_id", ""),
            "urgency": context.get("urgency", "Normal"),
            "requested_at": context.get("requested_at", datetime.now().isoformat()),
            "requested_format": context.get("requested_format", "summary")
        }

        # Parse scope
        scope_info = {
            "time_range": "all_files" if task_ticket.get("scan_all") else "date_range",
            "start_date": task_ticket.get("start_date").isoformat() if task_ticket.get("start_date") else None,
            "end_date": task_ticket.get("end_date").isoformat() if task_ticket.get("end_date") else None,
            "workspace": str(self.workspace_path),
            "filter_type": "all" if task_ticket.get("scan_all") else "date_range"
        }

        return {
            "requester": requester_info,
            "request": request_info,
            "scope": scope_info
        }

    def _generate_statistics(
        self,
        parsed_files: List[Dict[str, Any]],
        files_indexed: int
    ) -> Dict[str, Any]:
        """Generate comprehensive statistics."""

        total_size = sum(f.get("size", 0) for f in parsed_files)

        return {
            "files_scanned": len(parsed_files),  # Could be more accurate
            "files_matched": len(parsed_files),
            "files_processed": len(parsed_files),
            "files_indexed": files_indexed,
            "files_failed": 0,  # Count from errors
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 3)
        }

    def _generate_breakdown_by_type(
        self,
        parsed_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate breakdown by file type."""

        breakdown = {}

        # Group by type
        by_type: Dict[str, List[Dict]] = {}
        for f in parsed_files:
            ftype = f.get("file_type", "other")
            if ftype not in by_type:
                by_type[ftype] = []
            by_type[ftype].append(f)

        # Generate stats for each type
        for file_type, files in by_type.items():
            type_stats = {
                "count": len(files)
            }

            if file_type == "code":
                total_lines = 0
                total_functions = 0
                total_classes = 0
                languages = set()

                for f in files:
                    parse_result = f.get("parse_result", {})
                    details = parse_result.get("details", {})

                    total_lines += details.get("lines_of_code", 0)
                    total_functions += len(details.get("functions", []))
                    total_classes += len(details.get("classes", []))

                    # Infer language from extension
                    ext = Path(f.get("name", "")).suffix
                    if ext == ".py":
                        languages.add("python")

                type_stats.update({
                    "languages": list(languages),
                    "total_lines": total_lines,
                    "functions": total_functions,
                    "classes": total_classes
                })

            elif file_type == "data":
                formats = set()
                total_rows = 0
                total_cols = 0

                for f in files:
                    parse_result = f.get("parse_result", {})
                    details = parse_result.get("details", {})

                    ext = Path(f.get("name", "")).suffix.lstrip(".")
                    formats.add(ext)
                    total_rows += details.get("rows", 0)
                    total_cols = max(total_cols, details.get("columns", 0))

                type_stats.update({
                    "formats": list(formats),
                    "total_rows": total_rows,
                    "total_columns": total_cols
                })

            elif file_type == "notebook":
                total_cells = 0
                code_cells = 0
                markdown_cells = 0

                for f in files:
                    parse_result = f.get("parse_result", {})
                    details = parse_result.get("details", {})

                    total_cells += details.get("total_cells", 0)
                    code_cells += details.get("code_cells", 0)
                    markdown_cells += details.get("markdown_cells", 0)

                type_stats.update({
                    "total_cells": total_cells,
                    "code_cells": code_cells,
                    "markdown_cells": markdown_cells
                })

            elif file_type == "note":
                formats = set()
                total_words = 0

                for f in files:
                    parse_result = f.get("parse_result", {})
                    details = parse_result.get("details", {})

                    ext = Path(f.get("name", "")).suffix.lstrip(".")
                    formats.add(ext)
                    total_words += details.get("words", 0)

                type_stats.update({
                    "formats": list(formats),
                    "total_words": total_words
                })

            breakdown[file_type] = type_stats

        return breakdown

    def _generate_enhanced_file_list(
        self,
        parsed_files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate enhanced file list with detailed analysis."""

        enhanced = []

        for f in parsed_files:
            parse_result = f.get("parse_result", {})
            details = parse_result.get("details", {})

            file_info = {
                "path": f.get("path", ""),
                "name": f.get("name", ""),
                "type": f.get("file_type", "other"),
                "size_bytes": f.get("size", 0),
                "modified": f.get("modified").isoformat() if isinstance(f.get("modified"), datetime) else f.get("modified"),
                "created": None,  # Not available in current implementation
                "author": "unknown",

                "analysis": {
                    "summary": parse_result.get("summary", ""),
                    "details": details,
                    "content_preview": parse_result.get("content_preview", "")[:200],
                    "tags": self._generate_tags(f),
                    "rag_indexed": parse_result.get("success", True),
                    "rag_id": None  # Would need to track this from RAG manager
                }
            }

            # Add type-specific fields
            if f.get("file_type") == "code":
                file_info["analysis"]["language"] = "python"  # Could be more sophisticated
                file_info["analysis"]["complexity"] = self._estimate_complexity(details)
            elif f.get("file_type") == "data":
                file_info["analysis"]["format"] = Path(f.get("name", "")).suffix.lstrip(".")
            elif f.get("file_type") == "notebook":
                file_info["analysis"]["format"] = "jupyter"
                file_info["analysis"]["kernel"] = "python3"
                file_info["analysis"]["executed"] = False
                file_info["analysis"]["has_outputs"] = False
            elif f.get("file_type") == "note":
                file_info["analysis"]["format"] = Path(f.get("name", "")).suffix.lstrip(".")

            enhanced.append(file_info)

        return enhanced

    def _generate_tags(self, file_data: Dict[str, Any]) -> List[str]:
        """Generate tags for a file based on its content."""
        tags = []

        file_type = file_data.get("file_type", "")
        name = file_data.get("name", "").lower()
        parse_result = file_data.get("parse_result", {})
        details = parse_result.get("details", {})

        # Type-based tags
        if file_type == "code":
            tags.append("code")
            if "test" in name:
                tags.append("test")
            if "analysis" in name or "analyze" in name:
                tags.append("analysis")

            # Check imports for common libraries
            imports = details.get("imports", [])
            if "pandas" in imports:
                tags.append("pandas")
            if "numpy" in imports:
                tags.append("numpy")

        elif file_type == "data":
            tags.append("data")
            if "sensor" in name or "temperature" in name:
                tags.append("sensor-data")
            if details.get("columns"):
                if "timestamp" in str(details.get("column_names", [])).lower():
                    tags.append("timeseries")

        elif file_type == "notebook":
            tags.append("notebook")
            tags.append("jupyter")
            if "analysis" in name:
                tags.append("analysis")
            if "visualization" in name or "viz" in name:
                tags.append("visualization")

        elif file_type == "note":
            tags.append("documentation")
            if "research" in name:
                tags.append("research")
            if "readme" in name.lower():
                tags.append("readme")

        return tags

    def _estimate_complexity(self, details: Dict[str, Any]) -> str:
        """Estimate code complexity based on metrics."""
        functions = len(details.get("functions", []))
        classes = len(details.get("classes", []))
        lines = details.get("lines_of_code", 0)

        total_items = functions + classes

        if total_items == 0 or lines < 50:
            return "low"
        elif total_items < 5 and lines < 200:
            return "medium"
        else:
            return "high"

    def _generate_insights(self, parsed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from processed files."""

        if not parsed_files:
            return {
                "highlights": [],
                "activity_summary": "No activity found",
                "notable_changes": [],
                "recommendations": []
            }

        highlights = []
        notable_changes = []
        recommendations = []

        # Generate highlights
        for f in parsed_files:
            file_type = f.get("file_type", "")
            name = f.get("name", "")
            parse_result = f.get("parse_result", {})
            summary = parse_result.get("summary", "")

            if file_type == "code":
                highlights.append(f"New code added ({name})")
                notable_changes.append({
                    "type": "code",
                    "file": name,
                    "description": summary,
                    "impact": "high"
                })
                recommendations.append(f"Review {name} implementation")
            elif file_type == "data":
                highlights.append(f"Data file updated ({name})")
                notable_changes.append({
                    "type": "data",
                    "file": name,
                    "description": summary,
                    "impact": "medium"
                })
            elif file_type == "notebook":
                highlights.append(f"Analysis notebook created ({name})")
                recommendations.append(f"Execute {name} to generate visualizations")
            elif file_type == "note":
                highlights.append(f"Documentation updated ({name})")

        # Generate activity summary
        type_counts = {}
        for f in parsed_files:
            ftype = f.get("file_type", "other")
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

        if "code" in type_counts and type_counts["code"] > 0:
            activity_summary = "Active development with code changes"
        elif "data" in type_counts:
            activity_summary = "Data collection and processing activity"
        elif "note" in type_counts:
            activity_summary = "Documentation updates"
        else:
            activity_summary = "General workspace activity"

        return {
            "highlights": highlights[:10],  # Limit to 10
            "activity_summary": activity_summary,
            "notable_changes": notable_changes[:5],  # Limit to 5
            "recommendations": recommendations[:5]  # Limit to 5
        }

    def _get_rag_status_comprehensive(self) -> Dict[str, Any]:
        """Get comprehensive RAG status."""
        try:
            base_status = self.rag_manager.get_collection_info()
            return {
                "enabled": True,
                "collection_name": base_status.get("collection_name", "local_rag"),
                "total_documents": base_status.get("points_count", 0),
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "vector_count": base_status.get("vectors_count", 0),
                "index_updated": True,
                "searchable": True,
                "top_similar_queries": []  # Could track popular queries
            }
        except Exception as e:
            logger.error(f"Error getting RAG status: {e}")
            return {
                "enabled": False,
                "error": str(e)
            }

    def _generate_warnings(self, parsed_files: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate warnings based on file analysis."""
        warnings = []

        # Check for large files
        large_files = [f for f in parsed_files if f.get("size", 0) > 10 * 1024 * 1024]
        if large_files:
            warnings.append({
                "level": "info",
                "message": f"Found {len(large_files)} large files (>10MB)",
                "affected_files": [f.get("name") for f in large_files]
            })

        return warnings

    def _generate_suggested_actions(self, parsed_files: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate suggested follow-up actions."""
        actions = [
            {
                "action": "semantic_search",
                "description": "Search for specific topics in the workspace",
                "example": "Find all code related to 'data processing'"
            }
        ]

        # Add specific suggestions based on content
        has_code = any(f.get("file_type") == "code" for f in parsed_files)
        has_notebook = any(f.get("file_type") == "notebook" for f in parsed_files)

        if has_code:
            actions.append({
                "action": "deep_dive",
                "description": "Get detailed analysis of specific code file",
                "example": "Analyze [filename].py in detail"
            })

        if has_notebook:
            actions.append({
                "action": "execute_notebook",
                "description": "Run notebook to see results",
                "example": "Execute notebook and show outputs"
            })

        actions.append({
            "action": "compare",
            "description": "Compare with previous time period",
            "example": "Show changes from last week"
        })

        return actions

    def _build_error_result(
        self,
        task_ticket: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """Build error result."""
        return {
            "task_id": task_ticket.get("task_id", "unknown"),
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "message": f"Task failed: {error}",
            "context": self._build_context(task_ticket, context),
            "errors": [error],
            "performance": {
                "processing_time_seconds": processing_time
            },
            "metadata": {
                "agent_version": self.agent_version,
                "schema_version": "1.0"
            }
        }

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
