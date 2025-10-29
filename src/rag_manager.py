"""
RAG Manager Module for SignalMesh Local Agent.

This module provides RAG (Retrieval-Augmented Generation) functionality using
Qdrant vector database and sentence transformers for semantic search.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


class SearchResult(BaseModel):
    """Result from a semantic search query."""

    file_path: str = Field(..., description="Path to the matched file")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    content_preview: str = Field(default="", description="Content preview")


class RAGManager:
    """
    Manager for RAG operations using Qdrant and sentence transformers.

    Provides functionality to index files with semantic embeddings and
    perform similarity-based search.
    """

    def __init__(
        self,
        storage_path: str = "./data/qdrant_storage",
        collection_name: str = "local_rag",
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> None:
        """
        Initialize the RAG Manager.

        Args:
            storage_path: Path for Qdrant local storage
            collection_name: Name of the vector collection
            embedding_model: Name of the sentence transformer model

        Example:
            >>> rag = RAGManager()
            >>> rag.index_file("/path/to/file.py", "print('hello')", {...})
        """
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing RAG Manager with storage: {self.storage_path}")

        # Initialize Qdrant client in local mode
        self.client = QdrantClient(path=str(self.storage_path))

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """
        Initialize the Qdrant collection.

        Creates the collection if it doesn't exist, or verifies it if it does.
        """
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self.embedding_model.encode(text, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def index_file(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Index a file in the vector store.

        Args:
            file_path: Path to the file
            content: Content to index
            metadata: Additional metadata to store

        Returns:
            ID of the indexed point

        Example:
            >>> rag.index_file(
            ...     "/path/to/file.py",
            ...     "def hello(): print('world')",
            ...     {"file_type": "code", "size": 1024}
            ... )
        """
        try:
            # Generate unique ID based on file path
            point_id = str(uuid4())

            # Generate embedding
            embedding = self._generate_embedding(content)

            # Prepare payload
            payload = {
                "file_path": file_path,
                "content_preview": content[:500],  # Store preview
                **(metadata or {})
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            # Upsert to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(f"Indexed file: {file_path} (ID: {point_id})")
            return point_id

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            raise

    def index_multiple_files(
        self,
        files: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Index multiple files in batch.

        Args:
            files: List of dicts with 'file_path', 'content', and 'metadata' keys

        Returns:
            List of indexed point IDs

        Example:
            >>> files = [
            ...     {
            ...         "file_path": "/path/to/file1.py",
            ...         "content": "print('hello')",
            ...         "metadata": {"file_type": "code"}
            ...     },
            ...     ...
            ... ]
            >>> rag.index_multiple_files(files)
        """
        try:
            points = []

            for file_data in files:
                file_path = file_data.get("file_path")
                content = file_data.get("content", "")
                metadata = file_data.get("metadata", {})

                if not file_path or not content:
                    logger.warning(f"Skipping file with missing data: {file_data}")
                    continue

                point_id = str(uuid4())
                embedding = self._generate_embedding(content)

                payload = {
                    "file_path": file_path,
                    "content_preview": content[:500],
                    **metadata
                }

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )

                points.append(point)

            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Indexed {len(points)} files in batch")

            return [p.id for p in points]

        except Exception as e:
            logger.error(f"Error in batch indexing: {e}")
            raise

    def search_similar(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar content using semantic search.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of search results as dictionaries

        Example:
            >>> results = rag.search_similar("machine learning code", limit=5)
            >>> for result in results:
            ...     print(result['file_path'], result['score'])
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )

            # Format results
            results = []
            for hit in search_results:
                result = SearchResult(
                    file_path=hit.payload.get("file_path", ""),
                    score=hit.score,
                    metadata={
                        k: v for k, v in hit.payload.items()
                        if k not in ["file_path", "content_preview"]
                    },
                    content_preview=hit.payload.get("content_preview", "")
                )
                results.append(result.model_dump())

            logger.debug(f"Search query '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def delete_file(self, file_path: str) -> int:
        """
        Delete a file from the index.

        Args:
            file_path: Path to the file to delete

        Returns:
            Number of points deleted
        """
        try:
            # Search for points with this file path
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "file_path",
                            "match": {"value": file_path}
                        }
                    ]
                },
                limit=100
            )

            point_ids = [point.id for point in results[0]]

            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                logger.info(f"Deleted {len(point_ids)} points for file: {file_path}")
                return len(point_ids)

            return 0

        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Dictionary with collection statistics

        Example:
            >>> info = rag.get_collection_info()
            >>> print(f"Total files indexed: {info['points_count']}")
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    def clear_collection(self) -> None:
        """
        Clear all data from the collection.

        Warning: This will delete all indexed files!
        """
        try:
            logger.warning(f"Clearing all data from collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            logger.info("Collection cleared and recreated")

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
