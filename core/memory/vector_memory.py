"""
Vector Memory - Semantic memory using vector embeddings for context retrieval
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from core.memory.cache_eviction import CacheEvictionManager, EvictionPolicy


@dataclass
class MemoryEntry:
    """Represents a memory entry with embeddings"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = None
    relevance_score: float = 0.0
    access_count: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class VectorMemory:
    """Manages semantic memory using vector embeddings"""
    
    def __init__(self, project_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.project_name = project_name
        self.persist_dir = Path(f"persistence/vector_memory/{project_name}")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collections for different memory types
        self.collections = {
            "code": self._get_or_create_collection("code_memory"),
            "decisions": self._get_or_create_collection("decision_memory"),
            "errors": self._get_or_create_collection("error_memory"),
            "context": self._get_or_create_collection("context_memory"),
            "patterns": self._get_or_create_collection("pattern_memory")
        }
        
        # Memory statistics
        self.stats = {
            "total_memories": 0,
            "retrievals": 0,
            "hits": 0
        }

        # Cache eviction manager to prevent memory leaks
        self.cache_manager = CacheEvictionManager(
            max_entries=10000,  # Max 10k memories per collection
            max_size_bytes=500 * 1024 * 1024,  # 500 MB max
            policy=EvictionPolicy.HYBRID,  # LRU + TTL
            default_ttl_seconds=3600 * 24 * 30,  # 30 days default
            eviction_threshold=0.85  # Evict at 85% capacity
        )

        self._load_stats()
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"project": self.project_name}
            )
    
    def _load_stats(self):
        """Load memory statistics"""
        stats_file = self.persist_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                self.stats = json.load(f)
    
    def _save_stats(self):
        """Save memory statistics"""
        stats_file = self.persist_dir / "stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)
    
    def store_memory(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[Dict] = None,
        force_unique: bool = False
    ) -> str:
        """Store a memory with its embedding"""
        
        if memory_type not in self.collections:
            raise ValueError(f"Unknown memory type: {memory_type}")
        
        # Generate ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        memory_id = f"{memory_type}_{content_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check for duplicates if force_unique
        if force_unique:
            similar = self.search_memories(content, memory_type, top_k=1)
            if similar and similar[0].relevance_score > 0.95:
                logger.info(f"Duplicate memory detected, skipping: {memory_id}")
                return similar[0].id
        
        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "type": memory_type,
            "project": self.project_name,
            "length": len(content)
        })
        
        # Store in ChromaDB
        collection = self.collections[memory_type]
        collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
        
        self.stats["total_memories"] += 1
        self._save_stats()
        
        logger.info(f"Stored {memory_type} memory: {memory_id}")
        return memory_id
    
    def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_relevance: float = 0.0
    ) -> List[MemoryEntry]:
        """Search for relevant memories"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in specified collection or all
        if memory_type:
            collections = [self.collections[memory_type]]
        else:
            collections = self.collections.values()
        
        all_results = []
        
        for collection in collections:
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                # Process results
                if results['ids'][0]:
                    for i, id_ in enumerate(results['ids'][0]):
                        # Calculate relevance score (1 - distance for cosine similarity)
                        relevance = 1 - results['distances'][0][i]
                        
                        if relevance >= min_relevance:
                            entry = MemoryEntry(
                                id=id_,
                                content=results['documents'][0][i],
                                metadata=results['metadatas'][0][i],
                                relevance_score=relevance
                            )
                            all_results.append(entry)
            except Exception as e:
                logger.error(f"Error searching collection: {e}")
                continue
        
        # Sort by relevance and return top k
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        self.stats["retrievals"] += 1
        if all_results:
            self.stats["hits"] += 1
        self._save_stats()
        
        return all_results[:top_k]
    
    def get_context_window(
        self,
        current_context: str,
        window_size: int = 10,
        memory_types: List[str] = None
    ) -> List[MemoryEntry]:
        """Get relevant context window for current situation"""
        
        if memory_types is None:
            memory_types = ["context", "patterns", "decisions"]
        
        all_memories = []
        
        for memory_type in memory_types:
            if memory_type in self.collections:
                memories = self.search_memories(
                    current_context,
                    memory_type=memory_type,
                    top_k=window_size // len(memory_types)
                )
                all_memories.extend(memories)
        
        # Sort by relevance and recency
        all_memories.sort(
            key=lambda x: (x.relevance_score * 0.7 + 
                          (1 / (datetime.now() - datetime.fromisoformat(
                              x.metadata.get('timestamp', datetime.now().isoformat())
                          )).total_seconds() * 0.3)),
            reverse=True
        )
        
        return all_memories[:window_size]
    
    def consolidate_memories(
        self,
        memory_type: str,
        similarity_threshold: float = 0.9
    ) -> int:
        """Consolidate similar memories to reduce redundancy"""
        
        if memory_type not in self.collections:
            raise ValueError(f"Unknown memory type: {memory_type}")
        
        collection = self.collections[memory_type]
        
        # Get all memories
        all_data = collection.get()
        
        if not all_data['ids']:
            return 0
        
        consolidated_count = 0
        to_remove = set()
        
        # Find similar memories
        for i, id1 in enumerate(all_data['ids']):
            if id1 in to_remove:
                continue
                
            embedding1 = all_data['embeddings'][i]
            
            for j, id2 in enumerate(all_data['ids']):
                if i >= j or id2 in to_remove:
                    continue
                    
                embedding2 = all_data['embeddings'][j]
                
                # Calculate cosine similarity
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                
                if similarity > similarity_threshold:
                    # Merge metadata
                    merged_metadata = all_data['metadatas'][i].copy()
                    merged_metadata.update(all_data['metadatas'][j])
                    merged_metadata['consolidated'] = True
                    merged_metadata['original_ids'] = [id1, id2]
                    
                    # Keep the longer content
                    if len(all_data['documents'][i]) >= len(all_data['documents'][j]):
                        collection.update(
                            ids=[id1],
                            metadatas=[merged_metadata]
                        )
                        to_remove.add(id2)
                    else:
                        collection.update(
                            ids=[id2],
                            metadatas=[merged_metadata]
                        )
                        to_remove.add(id1)
                        break
                    
                    consolidated_count += 1
        
        # Remove consolidated memories
        if to_remove:
            collection.delete(ids=list(to_remove))
        
        logger.info(f"Consolidated {consolidated_count} memories in {memory_type}")
        return consolidated_count
    
    def export_memories(self, output_path: Path) -> bool:
        """Export all memories for backup or transfer"""
        
        try:
            export_data = {}
            
            for name, collection in self.collections.items():
                data = collection.get()
                export_data[name] = {
                    "ids": data['ids'],
                    "documents": data['documents'],
                    "metadatas": data['metadatas'],
                    "embeddings": data['embeddings']
                }
            
            # Save as pickle for preserving numpy arrays
            with open(output_path, "wb") as f:
                pickle.dump(export_data, f)
            
            logger.info(f"Exported memories to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export memories: {e}")
            return False
    
    def import_memories(self, input_path: Path) -> bool:
        """Import memories from backup"""
        
        try:
            with open(input_path, "rb") as f:
                import_data = pickle.load(f)
            
            for name, data in import_data.items():
                if name in self.collections:
                    collection = self.collections[name]
                    
                    # Clear existing data
                    collection.delete(where={})
                    
                    # Import new data
                    if data['ids']:
                        collection.add(
                            ids=data['ids'],
                            embeddings=data['embeddings'],
                            documents=data['documents'],
                            metadatas=data['metadatas']
                        )
            
            logger.info(f"Imported memories from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import memories: {e}")
            return False
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""

        stats = self.stats.copy()

        # Add collection-specific stats
        for name, collection in self.collections.items():
            count = collection.count()
            stats[f"{name}_count"] = count

        # Calculate hit rate
        if stats["retrievals"] > 0:
            stats["hit_rate"] = stats["hits"] / stats["retrievals"]
        else:
            stats["hit_rate"] = 0

        # Add cache eviction stats
        cache_stats = self.cache_manager.get_stats()
        stats["cache_evictions"] = cache_stats["evictions"]
        stats["cache_size_mb"] = cache_stats["size_mb"]
        stats["cache_hit_rate"] = cache_stats["hit_rate"]

        return stats

    def evict_old_memories(
        self,
        memory_type: str,
        older_than_days: int = 90,
        keep_high_value: bool = True
    ) -> int:
        """
        Evict old memories to prevent unbounded growth

        Args:
            memory_type: Type of memory to evict from
            older_than_days: Evict memories older than this
            keep_high_value: If True, preserve frequently accessed memories

        Returns:
            Number of memories evicted
        """
        if memory_type not in self.collections:
            raise ValueError(f"Unknown memory type: {memory_type}")

        collection = self.collections[memory_type]
        cutoff_date = datetime.now() - timedelta(days=older_than_days)

        # Get all memories
        all_data = collection.get(include=['metadatas'])

        if not all_data['ids']:
            return 0

        to_remove = []

        for i, memory_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][i]

            # Parse timestamp
            try:
                timestamp_str = metadata.get('timestamp', datetime.now().isoformat())
                timestamp = datetime.fromisoformat(timestamp_str)
            except:
                # If no valid timestamp, assume old
                timestamp = datetime.min

            # Check age
            if timestamp < cutoff_date:
                # If keeping high-value, check access count
                if keep_high_value:
                    access_count = metadata.get('access_count', 0)
                    if access_count > 10:  # Preserve frequently used
                        continue

                to_remove.append(memory_id)

        # Delete old memories
        if to_remove:
            collection.delete(ids=to_remove)
            logger.info(f"Evicted {len(to_remove)} old memories from {memory_type}")

        return len(to_remove)

    def evict_by_collection_size(
        self,
        memory_type: str,
        max_entries: int = 5000
    ) -> int:
        """
        Evict memories when collection exceeds max size

        Uses LRU policy to remove least recently accessed entries.

        Args:
            memory_type: Type of memory to evict from
            max_entries: Maximum entries to keep

        Returns:
            Number of memories evicted
        """
        if memory_type not in self.collections:
            raise ValueError(f"Unknown memory type: {memory_type}")

        collection = self.collections[memory_type]
        current_count = collection.count()

        if current_count <= max_entries:
            return 0

        # Get all memories with metadata
        all_data = collection.get(include=['metadatas'])

        # Sort by last_accessed (LRU)
        memory_access = []
        for i, memory_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][i]

            try:
                last_accessed_str = metadata.get('last_accessed', metadata.get('timestamp', datetime.min.isoformat()))
                last_accessed = datetime.fromisoformat(last_accessed_str)
            except:
                last_accessed = datetime.min

            memory_access.append((memory_id, last_accessed))

        # Sort by access time (oldest first)
        memory_access.sort(key=lambda x: x[1])

        # Evict oldest entries
        to_evict = current_count - max_entries
        to_remove = [memory_id for memory_id, _ in memory_access[:to_evict]]

        if to_remove:
            collection.delete(ids=to_remove)
            logger.info(f"Evicted {len(to_remove)} memories from {memory_type} (LRU policy)")

        return len(to_remove)

    def auto_evict_all_collections(
        self,
        max_entries_per_collection: int = 5000,
        max_age_days: int = 90
    ) -> Dict[str, int]:
        """
        Automatically evict from all collections

        Runs both age-based and size-based eviction.

        Returns:
            Dict mapping collection name to eviction count
        """
        eviction_results = {}

        for memory_type in self.collections.keys():
            # Age-based eviction
            age_evicted = self.evict_old_memories(
                memory_type,
                older_than_days=max_age_days,
                keep_high_value=True
            )

            # Size-based eviction
            size_evicted = self.evict_by_collection_size(
                memory_type,
                max_entries=max_entries_per_collection
            )

            total_evicted = age_evicted + size_evicted
            eviction_results[memory_type] = total_evicted

            if total_evicted > 0:
                logger.info(f"Collection {memory_type}: evicted {total_evicted} memories "
                           f"({age_evicted} by age, {size_evicted} by size)")

        return eviction_results
    
    def clear_memory_type(self, memory_type: str) -> bool:
        """Clear all memories of a specific type"""
        
        if memory_type not in self.collections:
            raise ValueError(f"Unknown memory type: {memory_type}")
        
        try:
            collection = self.collections[memory_type]
            collection.delete(where={})
            
            logger.info(f"Cleared all {memory_type} memories")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear {memory_type} memories: {e}")
            return False