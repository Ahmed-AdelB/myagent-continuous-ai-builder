"""
Memory Orchestrator - GPT-5 Recommendation #4
Unified Memory Architecture with Bidirectional Links

Unifies Project Ledger, Error Knowledge Graph, and Vector Memory
with cross-referencing capabilities and semantic indexing.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from core.memory.project_ledger import ProjectLedger
from core.memory.error_knowledge_graph import ErrorKnowledgeGraph
from core.memory.vector_memory import VectorMemory

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory entries"""
    PROJECT_EVENT = "project_event"
    ERROR_PATTERN = "error_pattern"
    CODE_CHANGE = "code_change"
    DESIGN_DECISION = "design_decision"
    TEST_RESULT = "test_result"
    PERFORMANCE_METRIC = "performance_metric"
    USER_FEEDBACK = "user_feedback"
    LEARNING_INSIGHT = "learning_insight"


class LinkType(Enum):
    """Types of bidirectional links between memory entries"""
    CAUSED_BY = "caused_by"
    RESOLVED_BY = "resolved_by"
    RELATED_TO = "related_to"
    IMPLEMENTS = "implements"
    TESTS = "tests"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    EVOLVED_FROM = "evolved_from"


@dataclass
class MemoryEntry:
    """Unified memory entry with cross-referencing"""
    id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    timestamp: datetime
    source_system: str  # Which memory system created this
    links: Dict[str, List[str]]  # LinkType -> list of linked entry IDs
    tags: Set[str]
    confidence_score: float = 1.0


@dataclass
class MemoryQueryResult:
    """Result of memory query with relevance scoring"""
    entries: List[MemoryEntry]
    query_embedding: List[float]
    similarity_scores: List[float]
    total_results: int
    query_time_ms: float


class MemoryOrchestrator:
    """
    Unified Memory Orchestrator

    Provides centralized access to all memory systems with:
    - Semantic search across all memory types
    - Bidirectional linking between entries
    - Cross-system correlation
    - Unified query interface
    - Automatic knowledge distillation
    """

    def __init__(self, project_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.project_name = project_name
        self.embedding_model_name = embedding_model

        # Initialize subsystems
        self.project_ledger = ProjectLedger(project_name)
        self.error_graph = ErrorKnowledgeGraph()  # ErrorKnowledgeGraph doesn't take parameters
        self.vector_memory = VectorMemory(project_name)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB for unified semantic search
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=f"persistence/memory_orchestrator_{project_name}",
        ))

        self.unified_collection = self.chroma_client.get_or_create_collection(
            name=f"unified_memory_{project_name}",
            metadata={"hnsw:space": "cosine"}
        )

        # Memory entry cache
        self.memory_cache: Dict[str, MemoryEntry] = {}

        logger.info(f"Memory Orchestrator initialized for project: {project_name}")

    async def store_memory(
        self,
        memory_type: MemoryType,
        content: str,
        metadata: Dict[str, Any],
        source_system: str,
        tags: Set[str] = None,
        auto_link: bool = True
    ) -> str:
        """
        Store a new memory entry with automatic linking

        Args:
            memory_type: Type of memory entry
            content: Text content of the memory
            metadata: Additional structured metadata
            source_system: Which system created this entry
            tags: Optional tags for categorization
            auto_link: Whether to automatically create links to similar entries

        Returns:
            str: Unique ID of the stored memory entry
        """
        entry_id = str(uuid.uuid4())
        tags = tags or set()

        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()

        # Create memory entry
        memory_entry = MemoryEntry(
            id=entry_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata,
            embedding=embedding,
            timestamp=datetime.now(),
            source_system=source_system,
            links={},
            tags=tags
        )

        # Store in appropriate subsystem
        await self._store_in_subsystem(memory_entry)

        # Store in unified collection
        self.unified_collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{
                "memory_type": memory_type.value,
                "source_system": source_system,
                "timestamp": memory_entry.timestamp.isoformat(),
                "tags": "|".join(tags),
                **{k: str(v) for k, v in metadata.items()}
            }],
            ids=[entry_id]
        )

        # Cache entry
        self.memory_cache[entry_id] = memory_entry

        # Auto-link if requested
        if auto_link:
            await self._create_automatic_links(memory_entry)

        logger.debug(f"Stored memory entry {entry_id} of type {memory_type.value}")
        return entry_id

    async def query_memory(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        include_links: bool = True,
        time_range: Tuple[datetime, datetime] = None
    ) -> MemoryQueryResult:
        """
        Query memory across all systems with semantic search

        Args:
            query: Natural language query
            memory_types: Filter by specific memory types
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            include_links: Whether to include linked entries
            time_range: Filter by time range (start, end)

        Returns:
            MemoryQueryResult with matching entries and metadata
        """
        start_time = datetime.now()

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build filter criteria
        where_filter = {}
        if memory_types:
            where_filter["memory_type"] = {"$in": [mt.value for mt in memory_types]}

        if time_range:
            start_iso, end_iso = time_range[0].isoformat(), time_range[1].isoformat()
            where_filter["timestamp"] = {"$gte": start_iso, "$lte": end_iso}

        # Query unified collection
        results = self.unified_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 2,  # Get more to filter by threshold
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        # Process results
        matching_entries = []
        similarity_scores = []

        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            # Convert distance to similarity score (cosine distance -> similarity)
            similarity = 1.0 - distance

            if similarity >= similarity_threshold:
                # Reconstruct memory entry
                entry = await self._reconstruct_memory_entry(doc_id, document, metadata)
                if entry:
                    matching_entries.append(entry)
                    similarity_scores.append(similarity)

                    # Include linked entries if requested
                    if include_links and len(matching_entries) < limit:
                        linked_entries = await self._get_linked_entries(entry, similarity_threshold * 0.8)
                        for linked_entry in linked_entries:
                            if len(matching_entries) < limit:
                                matching_entries.append(linked_entry)
                                similarity_scores.append(similarity * 0.9)  # Slightly lower score for linked entries

            if len(matching_entries) >= limit:
                break

        # Calculate query time
        query_time = (datetime.now() - start_time).total_seconds() * 1000

        return MemoryQueryResult(
            entries=matching_entries[:limit],
            query_embedding=query_embedding,
            similarity_scores=similarity_scores[:limit],
            total_results=len(results["ids"][0]),
            query_time_ms=query_time
        )

    async def create_link(
        self,
        source_entry_id: str,
        target_entry_id: str,
        link_type: LinkType,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Create a bidirectional link between two memory entries

        Args:
            source_entry_id: ID of the source entry
            target_entry_id: ID of the target entry
            link_type: Type of link to create
            confidence: Confidence score for the link (0.0 to 1.0)
            metadata: Additional metadata for the link

        Returns:
            bool: True if link was created successfully
        """
        try:
            # Get entries from cache or storage
            source_entry = await self._get_memory_entry(source_entry_id)
            target_entry = await self._get_memory_entry(target_entry_id)

            if not source_entry or not target_entry:
                logger.error(f"Failed to find entries for link: {source_entry_id} -> {target_entry_id}")
                return False

            # Create forward link
            link_key = link_type.value
            if link_key not in source_entry.links:
                source_entry.links[link_key] = []

            link_data = {
                "target_id": target_entry_id,
                "confidence": confidence,
                "created": datetime.now().isoformat()
            }
            if metadata:
                link_data["metadata"] = metadata

            source_entry.links[link_key].append(json.dumps(link_data))

            # Create reverse link
            reverse_link_key = self._get_reverse_link_type(link_type)
            if reverse_link_key:
                if reverse_link_key not in target_entry.links:
                    target_entry.links[reverse_link_key] = []

                reverse_link_data = {
                    "target_id": source_entry_id,
                    "confidence": confidence,
                    "created": datetime.now().isoformat()
                }
                if metadata:
                    reverse_link_data["metadata"] = metadata

                target_entry.links[reverse_link_key].append(json.dumps(reverse_link_data))

            # Update cache and storage
            self.memory_cache[source_entry_id] = source_entry
            self.memory_cache[target_entry_id] = target_entry

            await self._update_memory_entry_storage(source_entry)
            await self._update_memory_entry_storage(target_entry)

            logger.debug(f"Created link {link_type.value}: {source_entry_id} -> {target_entry_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create link: {e}")
            return False

    async def get_memory_graph(self, center_entry_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get a graph representation of memory connections

        Args:
            center_entry_id: ID of the central entry
            depth: How many levels of links to include

        Returns:
            Dict representation of the memory graph
        """
        visited = set()
        nodes = {}
        edges = []

        async def _traverse(entry_id: str, current_depth: int):
            if entry_id in visited or current_depth > depth:
                return

            visited.add(entry_id)
            entry = await self._get_memory_entry(entry_id)

            if not entry:
                return

            # Add node
            nodes[entry_id] = {
                "id": entry_id,
                "memory_type": entry.memory_type.value,
                "content": entry.content[:200] + "..." if len(entry.content) > 200 else entry.content,
                "timestamp": entry.timestamp.isoformat(),
                "source_system": entry.source_system,
                "tags": list(entry.tags)
            }

            # Traverse links
            for link_type, link_list in entry.links.items():
                for link_json in link_list:
                    link_data = json.loads(link_json)
                    target_id = link_data["target_id"]

                    # Add edge
                    edges.append({
                        "source": entry_id,
                        "target": target_id,
                        "link_type": link_type,
                        "confidence": link_data.get("confidence", 1.0)
                    })

                    # Recurse
                    await _traverse(target_id, current_depth + 1)

        await _traverse(center_entry_id, 0)

        return {
            "center_entry_id": center_entry_id,
            "nodes": nodes,
            "edges": edges,
            "depth": depth,
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }

    async def find_knowledge_patterns(self, pattern_type: str = "frequent_errors") -> List[Dict[str, Any]]:
        """
        Identify knowledge patterns across all memory systems

        Args:
            pattern_type: Type of pattern to find

        Returns:
            List of identified patterns
        """
        patterns = []

        if pattern_type == "frequent_errors":
            # Find frequently occurring error patterns
            error_entries = await self.query_memory(
                query="error failure bug issue",
                memory_types=[MemoryType.ERROR_PATTERN],
                limit=100,
                similarity_threshold=0.5
            )

            # Group by similarity
            error_clusters = self._cluster_similar_entries(error_entries.entries)

            for cluster in error_clusters:
                if len(cluster) >= 3:  # At least 3 similar errors
                    patterns.append({
                        "pattern_type": "frequent_error",
                        "frequency": len(cluster),
                        "representative_content": cluster[0].content,
                        "entry_ids": [e.id for e in cluster],
                        "confidence": sum(e.confidence_score for e in cluster) / len(cluster)
                    })

        elif pattern_type == "solution_success":
            # Find successful solution patterns
            success_entries = await self.query_memory(
                query="fixed resolved solved successful",
                memory_types=[MemoryType.PROJECT_EVENT, MemoryType.CODE_CHANGE],
                limit=100,
                similarity_threshold=0.6
            )

            for entry in success_entries.entries:
                # Find what this solution resolved
                linked_errors = await self._get_linked_entries_by_type(entry, LinkType.RESOLVED_BY)
                if linked_errors:
                    patterns.append({
                        "pattern_type": "solution_success",
                        "solution_content": entry.content,
                        "resolved_errors": [e.content for e in linked_errors],
                        "solution_id": entry.id,
                        "confidence": entry.confidence_score
                    })

        return patterns

    async def distill_knowledge(self, time_window_days: int = 30) -> Dict[str, Any]:
        """
        Distill and compress knowledge from the specified time window

        Args:
            time_window_days: Number of days to look back

        Returns:
            Dict containing distilled knowledge
        """
        # Get entries from the time window
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_window_days)

        all_entries = await self.query_memory(
            query="",
            limit=1000,
            similarity_threshold=0.0,
            time_range=(start_time, end_time)
        )

        # Categorize entries
        categorized = {
            "errors": [],
            "solutions": [],
            "insights": [],
            "patterns": []
        }

        for entry in all_entries.entries:
            if entry.memory_type == MemoryType.ERROR_PATTERN:
                categorized["errors"].append(entry)
            elif entry.memory_type in [MemoryType.CODE_CHANGE, MemoryType.DESIGN_DECISION]:
                categorized["solutions"].append(entry)
            elif entry.memory_type == MemoryType.LEARNING_INSIGHT:
                categorized["insights"].append(entry)

        # Find patterns
        categorized["patterns"] = await self.find_knowledge_patterns()

        # Create distilled summary
        distilled = {
            "time_window": f"{time_window_days} days",
            "period": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "total_entries": len(all_entries.entries),
            "categories": {k: len(v) for k, v in categorized.items()},
            "top_patterns": categorized["patterns"][:10],
            "distillation_timestamp": datetime.now().isoformat()
        }

        return distilled

    async def _store_in_subsystem(self, memory_entry: MemoryEntry):
        """Store memory entry in appropriate subsystem"""
        if memory_entry.memory_type == MemoryType.ERROR_PATTERN:
            # Store in error knowledge graph (using add_error, not add_error_pattern)
            error_node = self.error_graph.add_error(
                error_type=memory_entry.metadata.get("error_type", "unknown"),
                error_message=memory_entry.content,
                context=memory_entry.metadata.get("context", {})
            )

            # If there's a solution in metadata, add it too
            solution = memory_entry.metadata.get("solution")
            if solution:
                self.error_graph.add_solution(
                    error_id=error_node.id,
                    solution_type="fix",
                    description=solution,
                    code_changes=memory_entry.metadata.get("code_changes", {})
                )

        elif memory_entry.memory_type in [MemoryType.PROJECT_EVENT, MemoryType.CODE_CHANGE]:
            # Store in project ledger (synchronous method, don't await)
            self.project_ledger.record_decision(
                iteration=memory_entry.metadata.get("iteration_id", 0),
                agent=memory_entry.metadata.get("agent_name", "unknown"),
                decision_type=memory_entry.metadata.get("decision_type", "general"),
                description=memory_entry.content,
                rationale=memory_entry.metadata.get("rationale", "")
            )

        else:
            # Store in vector memory (using store_memory, not store_embedding)
            self.vector_memory.store_memory(
                memory_type=memory_entry.memory_type.value,
                content=memory_entry.content,
                metadata=memory_entry.metadata
            )

    async def _create_automatic_links(self, memory_entry: MemoryEntry):
        """Create automatic links based on content similarity and patterns"""
        # Find similar entries
        similar_results = await self.query_memory(
            query=memory_entry.content,
            memory_types=None,
            limit=5,
            similarity_threshold=0.8,
            include_links=False
        )

        for similar_entry in similar_results.entries:
            if similar_entry.id != memory_entry.id:
                await self.create_link(
                    memory_entry.id,
                    similar_entry.id,
                    LinkType.SIMILAR_TO,
                    confidence=0.8
                )

        # Create specific links based on content analysis
        if "error" in memory_entry.content.lower() and "fixed" in memory_entry.content.lower():
            # This might be a solution to an error
            error_results = await self.query_memory(
                query=memory_entry.content,
                memory_types=[MemoryType.ERROR_PATTERN],
                limit=3,
                similarity_threshold=0.7
            )

            for error_entry in error_results.entries:
                await self.create_link(
                    memory_entry.id,
                    error_entry.id,
                    LinkType.RESOLVED_BY,
                    confidence=0.7
                )

    async def _get_memory_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get memory entry by ID from cache or storage"""
        if entry_id in self.memory_cache:
            return self.memory_cache[entry_id]

        # Try to reconstruct from unified collection
        try:
            results = self.unified_collection.get(
                ids=[entry_id],
                include=["documents", "metadatas"]
            )

            if results["ids"] and len(results["ids"]) > 0:
                document = results["documents"][0]
                metadata = results["metadatas"][0]
                return await self._reconstruct_memory_entry(entry_id, document, metadata)

        except Exception as e:
            logger.error(f"Failed to get memory entry {entry_id}: {e}")

        return None

    async def _reconstruct_memory_entry(self, entry_id: str, document: str, metadata: Dict) -> Optional[MemoryEntry]:
        """Reconstruct memory entry from stored data"""
        try:
            memory_type = MemoryType(metadata.get("memory_type", "project_event"))
            timestamp = datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
            tags = set(metadata.get("tags", "").split("|")) if metadata.get("tags") else set()

            # Extract non-standard metadata
            entry_metadata = {
                k: v for k, v in metadata.items()
                if k not in ["memory_type", "source_system", "timestamp", "tags"]
            }

            return MemoryEntry(
                id=entry_id,
                memory_type=memory_type,
                content=document,
                metadata=entry_metadata,
                embedding=None,  # Will be regenerated if needed
                timestamp=timestamp,
                source_system=metadata.get("source_system", "unknown"),
                links={},  # Links stored separately
                tags=tags
            )

        except Exception as e:
            logger.error(f"Failed to reconstruct memory entry {entry_id}: {e}")
            return None

    async def _get_linked_entries(self, entry: MemoryEntry, min_confidence: float = 0.5) -> List[MemoryEntry]:
        """Get all entries linked to the given entry"""
        linked_entries = []

        for link_type, link_list in entry.links.items():
            for link_json in link_list:
                try:
                    link_data = json.loads(link_json)
                    if link_data.get("confidence", 1.0) >= min_confidence:
                        linked_entry = await self._get_memory_entry(link_data["target_id"])
                        if linked_entry:
                            linked_entries.append(linked_entry)
                except Exception as e:
                    logger.error(f"Error processing link: {e}")

        return linked_entries

    async def _get_linked_entries_by_type(self, entry: MemoryEntry, link_type: LinkType) -> List[MemoryEntry]:
        """Get entries linked by specific link type"""
        linked_entries = []

        link_list = entry.links.get(link_type.value, [])
        for link_json in link_list:
            try:
                link_data = json.loads(link_json)
                linked_entry = await self._get_memory_entry(link_data["target_id"])
                if linked_entry:
                    linked_entries.append(linked_entry)
            except Exception as e:
                logger.error(f"Error processing link: {e}")

        return linked_entries

    def _cluster_similar_entries(self, entries: List[MemoryEntry], similarity_threshold: float = 0.8) -> List[List[MemoryEntry]]:
        """Cluster entries by similarity"""
        if not entries:
            return []

        clusters = []
        used_entries = set()

        for i, entry in enumerate(entries):
            if entry.id in used_entries:
                continue

            cluster = [entry]
            used_entries.add(entry.id)

            # Find similar entries
            for j, other_entry in enumerate(entries[i+1:], i+1):
                if other_entry.id in used_entries:
                    continue

                # Calculate similarity if embeddings available
                if entry.embedding and other_entry.embedding:
                    similarity = self._calculate_similarity(entry.embedding, other_entry.embedding)
                    if similarity >= similarity_threshold:
                        cluster.append(other_entry)
                        used_entries.add(other_entry.id)

            clusters.append(cluster)

        return clusters

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0

    async def _update_memory_entry_storage(self, entry: MemoryEntry):
        """Update memory entry in storage"""
        try:
            # Update in unified collection
            self.unified_collection.update(
                ids=[entry.id],
                documents=[entry.content],
                metadatas=[{
                    "memory_type": entry.memory_type.value,
                    "source_system": entry.source_system,
                    "timestamp": entry.timestamp.isoformat(),
                    "tags": "|".join(entry.tags),
                    **{k: str(v) for k, v in entry.metadata.items()}
                }]
            )

            # Update subsystem storage
            await self._store_in_subsystem(entry)

        except Exception as e:
            logger.error(f"Failed to update memory entry storage: {e}")

    def _get_reverse_link_type(self, link_type: LinkType) -> Optional[str]:
        """Get the reverse link type for bidirectional linking"""
        reverse_map = {
            LinkType.CAUSED_BY: "causes",
            LinkType.RESOLVED_BY: "resolves",
            LinkType.RELATED_TO: LinkType.RELATED_TO.value,
            LinkType.IMPLEMENTS: "implemented_by",
            LinkType.TESTS: "tested_by",
            LinkType.DEPENDS_ON: "dependency_for",
            LinkType.SIMILAR_TO: LinkType.SIMILAR_TO.value,
            LinkType.EVOLVED_FROM: "evolved_to"
        }

        return reverse_map.get(link_type)

    async def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory orchestrator"""
        # Count entries by type
        type_counts = {}
        for memory_type in MemoryType:
            results = await self.query_memory(
                query="",
                memory_types=[memory_type],
                limit=1000,
                similarity_threshold=0.0
            )
            type_counts[memory_type.value] = results.total_results

        return {
            "project_name": self.project_name,
            "total_entries": sum(type_counts.values()),
            "entries_by_type": type_counts,
            "cached_entries": len(self.memory_cache),
            "embedding_model": self.embedding_model_name,
            "collection_info": self.unified_collection.count() if self.unified_collection else 0
        }