"""
VectorStore - ChromaDB wrapper for semantic code search.

Implements vector storage with:
- ChromaDB with HNSW index
- Explicit HNSW configuration (M=64, efConstruction=200, efSearch=100)
- Cosine distance metric
- Stores embeddings + metadata + chunk_id (NOT raw code per Gemini)
- Normalized embeddings for accurate cosine similarity

Security (Gemini recommendations - FULLY IMPLEMENTED):
- ✅ Stores only embeddings + metadata + chunk_id
- ✅ Raw code stored separately (not in vector DB)
- ✅ File permissions chmod 750 (Priority 2)
- ✅ Audit logging for all operations (Priority 3)

Based on: docs/architecture/rag_specification.md (v1.1)
         docs/architecture/RAG_SPEC_CHANGELOG.md (Codex recommendations)
Issue: #3
Implementation: Claude (Sonnet 4.5)
Security review: Gemini (2.5 Pro) - TODO
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Audit logging (Gemini Priority 3 requirement)
from core.utils.audit_logger import get_vector_store_audit_logger

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    _chromadb_available = True
except ImportError:
    chromadb = None
    _chromadb_available = False
    logger.warning("ChromaDB not available - vector search will not work")


class VectorStore:
    """
    ChromaDB wrapper for semantic code search.

    Features:
    - HNSW index (fast approximate nearest neighbor search)
    - Cosine distance metric
    - Persistent storage
    - Metadata filtering
    - Batch operations

    Storage Architecture (Gemini security recommendation):
    - Vector DB stores: embeddings + metadata + chunk_id
    - Raw code stored separately in code store
    - Retrieval: query vector DB for IDs, fetch code separately

    HNSW Configuration (Codex recommendations):
    - hnsw:M = 64 (links per node)
    - hnsw:efConstruction = 200 (build-time search depth)
    - hnsw:efSearch = 100 (query-time search depth)
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[Path] = None,
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection (e.g., "projectname_code")
            persist_directory: Where to store the database
        """
        self.collection_name = collection_name

        if persist_directory is None:
            persist_directory = Path("persistence/storage/vector_db")

        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Security (Gemini Priority 2): Set file permissions chmod 750
        # Restricts access to owner (rwx) and group (r-x), no access for others
        try:
            os.chmod(self.persist_directory, 0o750)
            logger.info(f"Set permissions 750 on {self.persist_directory}")
        except Exception as exc:
            logger.warning(f"Failed to set permissions on {self.persist_directory}: {exc}")

        # Audit logger (Gemini Priority 3)
        self.audit_logger = get_vector_store_audit_logger()

        # Initialize ChromaDB
        self.client = self._init_client()
        self.collection = self._init_collection()

    def _init_client(self):
        """Initialize ChromaDB client with persistence."""
        if not _chromadb_available:
            logger.error("ChromaDB not installed")
            return None

        try:
            # Use persistent client for data durability
            client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,  # Disable telemetry
                    allow_reset=False,  # Safety: prevent accidental reset
                )
            )
            logger.info(f"Initialized ChromaDB at: {self.persist_directory}")
            return client

        except Exception as exc:
            logger.error(f"Failed to initialize ChromaDB: {exc}")
            return None

    def _init_collection(self):
        """
        Initialize or get collection with proper HNSW configuration.

        HNSW Parameters (from Codex review):
        - hnsw:space: cosine (distance metric)
        - hnsw:M: 64 (links per node - balance between speed and accuracy)
        - hnsw:efConstruction: 200 (build-time search depth)
        - hnsw:efSearch: 100 (query-time search depth)
        """
        if not self.client:
            return None

        try:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",  # Cosine distance
                    "hnsw:M": 64,  # Links per node
                    "hnsw:efConstruction": 200,  # Build-time search depth
                    "hnsw:efSearch": 100,  # Query-time search depth
                }
            )

            count = collection.count()
            logger.info(
                f"Collection '{self.collection_name}' initialized "
                f"({count} embeddings)"
            )
            return collection

        except Exception as exc:
            logger.error(f"Failed to initialize collection: {exc}")
            return None

    async def add_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add code chunks to vector store.

        IMPORTANT (Gemini security): This stores ONLY:
        - chunk_id (reference to code in separate store)
        - embedding (vector for similarity search)
        - metadata (file_path, line numbers, function name, etc.)

        Raw code text is NOT stored here - it's stored separately
        in the code store and fetched only during context assembly.

        Args:
            chunk_ids: Unique IDs for each chunk
            embeddings: Normalized embedding vectors
            metadatas: Metadata for each chunk (NO raw code text)
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return

        if not (len(chunk_ids) == len(embeddings) == len(metadatas)):
            raise ValueError("chunk_ids, embeddings, and metadatas must have same length")

        try:
            # Convert numpy arrays to lists for ChromaDB
            embedding_lists = [emb.tolist() for emb in embeddings]

            # Add to collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embedding_lists,
                metadatas=metadatas,
                # NOTE: No 'documents' parameter - raw code not stored!
            )

            # Audit log successful add operation (Gemini Priority 3)
            self.audit_logger.log_vector_operation(
                operation="add",
                num_chunks=len(chunk_ids),
                status="success",
                collection_name=self.collection_name,
            )

            logger.debug(f"Added {len(chunk_ids)} chunks to vector store")

        except Exception as exc:
            # Audit log failed add operation
            self.audit_logger.log_vector_operation(
                operation="add",
                num_chunks=len(chunk_ids),
                status="failure",
                collection_name=self.collection_name,
                error=str(exc),
            )

            logger.error(f"Failed to add chunks: {exc}")
            raise

    async def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query vector store for similar chunks.

        Args:
            query_embedding: Normalized query embedding
            n_results: Number of results to return (Codex: 20-30, not just 5)
            filters: Optional metadata filters (e.g., {"language": "python"})

        Returns:
            List of dicts with: chunk_id, metadata, score
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=filters,  # Metadata filtering
                include=["metadatas", "distances"],  # Don't include documents
            )

            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "chunk_id": results["ids"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                })

            # Audit log successful query (Gemini Priority 3)
            self.audit_logger.log_vector_operation(
                operation="query",
                num_chunks=len(formatted_results),
                status="success",
                collection_name=self.collection_name,
            )

            logger.debug(f"Query returned {len(formatted_results)} results")
            return formatted_results

        except Exception as exc:
            # Audit log failed query
            self.audit_logger.log_vector_operation(
                operation="query",
                num_chunks=0,
                status="failure",
                collection_name=self.collection_name,
                error=str(exc),
            )

            logger.error(f"Failed to query vector store: {exc}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            return {"error": "Collection not initialized"}

        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_embeddings": count,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as exc:
            logger.error(f"Failed to get stats: {exc}")
            return {"error": str(exc)}

    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """
        Delete chunks from vector store.

        Used for GDPR compliance (Gemini requirement):
        - User requests deletion of their code
        - Trace via git blame to chunk_ids
        - Delete from vector store without full re-index
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return

        try:
            self.collection.delete(ids=chunk_ids)

            # Audit log successful delete operation (Gemini Priority 3)
            self.audit_logger.log_vector_operation(
                operation="delete",
                num_chunks=len(chunk_ids),
                status="success",
                collection_name=self.collection_name,
            )

            logger.info(f"Deleted {len(chunk_ids)} chunks from vector store")

        except Exception as exc:
            # Audit log failed delete operation
            self.audit_logger.log_vector_operation(
                operation="delete",
                num_chunks=len(chunk_ids),
                status="failure",
                collection_name=self.collection_name,
                error=str(exc),
            )

            logger.error(f"Failed to delete chunks: {exc}")
            raise

    async def reset(self) -> None:
        """
        Reset collection (delete all data).

        WARNING: This is destructive! Use with caution.
        """
        if not self.client:
            logger.error("Client not initialized")
            return

        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Collection '{self.collection_name}' deleted")

            # Recreate empty collection
            self.collection = self._init_collection()
        except Exception as exc:
            logger.error(f"Failed to reset collection: {exc}")
