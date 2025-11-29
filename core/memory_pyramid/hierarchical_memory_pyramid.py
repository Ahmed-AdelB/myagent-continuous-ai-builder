"""
Hierarchical Memory Pyramid - GPT-5 Priority 4
Implements scalable multi-tier learning and memory architecture.
"""

import asyncio
import json
import time
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from collections import defaultdict, deque
import os
import pickle
import sqlite3
from pathlib import Path

from ..observability.telemetry_engine import get_telemetry, LogLevel

class MemoryTier(Enum):
    WORKING = "working"        # Hot memory - immediate access
    SHORT_TERM = "short_term"  # Recent experiences - fast retrieval
    LONG_TERM = "long_term"    # Consolidated knowledge - structured
    ARCHIVE = "archive"        # Historical data - compressed

class MemoryType(Enum):
    EPISODIC = "episodic"      # Specific experiences/events
    SEMANTIC = "semantic"      # General knowledge/facts
    PROCEDURAL = "procedural"  # How-to knowledge/skills
    CONTEXTUAL = "contextual"  # Situational patterns
    USER_REQUEST = "user_request"
    IMPLEMENTATION = "implementation"
    DECISION = "decision"
    BUG_FIX = "bug_fix"
    OPTIMIZATION = "optimization"
    IMPROVEMENT = "improvement"
    ERROR = "error"
    SUCCESS = "success"

class MemoryPriority(Enum):
    CRITICAL = "critical"      # Must retain
    HIGH = "high"             # Important to retain
    MEDIUM = "medium"         # Moderately important
    LOW = "low"              # Can be discarded if needed

@dataclass
class MemoryNode:
    """Individual memory unit in the hierarchy"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    memory_type: MemoryType = MemoryType.EPISODIC
    priority: MemoryPriority = MemoryPriority.MEDIUM
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    tier: MemoryTier = MemoryTier.WORKING

    # Relationships
    parent_nodes: List[str] = field(default_factory=list)
    child_nodes: List[str] = field(default_factory=list)
    related_nodes: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    compression_ratio: float = 1.0
    importance: float = 0.5

    @property
    def compressed(self) -> bool:
        return self.compression_ratio > 1.0

    @property
    def last_accessed_at(self) -> datetime:
        return self.last_accessed

    @last_accessed_at.setter
    def last_accessed_at(self, value: datetime):
        self.last_accessed = value

    @property
    def created_at(self) -> datetime:
        return self.creation_time

    @created_at.setter
    def created_at(self, value: datetime):
        self.creation_time = value

    @property
    def is_deleted(self) -> bool:
        return False  # Basic implementation

    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def calculate_decayed_importance(self, decay_rate: float = 0.01) -> float:
        """Calculate importance with time decay"""
        now = datetime.now(timezone.utc)
        created = self.creation_time
        
        # Handle timezone mismatch
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
            
        time_diff = (now - created).total_seconds() / 3600  # Hours
        return self.importance * (1.0 / (1.0 + decay_rate * time_diff))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'priority': self.priority.value,
            'creation_time': self.creation_time.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'tier': self.tier.value,
            'parent_nodes': self.parent_nodes,
            'child_nodes': self.child_nodes,
            'related_nodes': self.related_nodes,
            'metadata': self.metadata,
            'embedding': self.embedding,
            'compression_ratio': self.compression_ratio,
            'importance': self.importance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        return cls(
            node_id=data['node_id'],
            content=data['content'],
            memory_type=MemoryType(data['memory_type']),
            priority=MemoryPriority(data['priority']),
            creation_time=datetime.fromisoformat(data['creation_time']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data['access_count'],
            tier=MemoryTier(data['tier']),
            parent_nodes=data['parent_nodes'],
            child_nodes=data['child_nodes'],
            related_nodes=data['related_nodes'],
            metadata=data['metadata'],
            embedding=data.get('embedding'),
            compression_ratio=data.get('compression_ratio', 1.0),
            importance=data.get('importance', 0.5)
        )

    def __lt__(self, other):
        if not isinstance(other, MemoryNode):
            return NotImplemented
        # Compare based on importance, then priority, then recency
        if self.importance != other.importance:
            return self.importance < other.importance
        if self.priority.value != other.priority.value:
            # This is simplified; ideally we'd map priority to int
            return self.priority.value < other.priority.value
        return self.creation_time < other.creation_time

@dataclass
class ConsolidationPattern:
    """Pattern for memory consolidation"""
    pattern_id: str
    source_tier: MemoryTier
    target_tier: MemoryTier
    conditions: Dict[str, Any]
    strategy: str
    compression_algorithm: str
    retention_policy: Dict[str, Any]

class HierarchicalMemoryPyramid:
    """
    Hierarchical Memory Pyramid System

    Implements GPT-5's recommendation for scalable multi-tier learning:
    - Working Memory: Hot access, immediate processing
    - Short-term Memory: Recent experiences, fast retrieval
    - Long-term Memory: Consolidated knowledge, structured storage
    - Archive Memory: Historical data, compressed storage

    Features:
    - Automatic tier promotion/demotion based on usage patterns
    - Memory consolidation and compression
    - Relationship tracking and graph-based retrieval
    - Adaptive capacity management
    - Pattern-based knowledge extraction
    """

    def __init__(self, storage_path: str = "./memory_pyramid", database_path: str = None, telemetry=None):
        self.telemetry = telemetry or get_telemetry()
        self.telemetry.register_component('hierarchical_memory_pyramid')

        # Storage configuration
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Database path override
        self.db_path_override = Path(database_path) if database_path else None

        # Memory tiers with capacity limits
        self.tier_capacities = {
            MemoryTier.WORKING: 1000,      # 1K nodes
            MemoryTier.SHORT_TERM: 10000,  # 10K nodes
            MemoryTier.LONG_TERM: 100000,  # 100K nodes
            MemoryTier.ARCHIVE: 1000000    # 1M nodes
        }

        # Memory storage by tier
        self.memory_tiers = {
            tier: {} for tier in MemoryTier
        }

        # Indices for fast retrieval
        self.type_index = defaultdict(set)      # type -> node_ids
        self.priority_index = defaultdict(set)   # priority -> node_ids
        self.temporal_index = defaultdict(list)  # date -> node_ids
        self.relationship_graph = defaultdict(set)  # node_id -> related_node_ids

        # Configuration
        self.consolidation_enabled = True
        self.compression_enabled = True
        self.auto_promotion_enabled = True

        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._background_thread = None

        # Performance tracking
        self.memory_metrics = {
            'total_nodes': 0,
            'tier_distributions': {tier.value: 0 for tier in MemoryTier},
            'consolidations_performed': 0,
            'promotions': 0,
            'demotions': 0,
            'compressions': 0
        }
        # Metrics
        self.metrics = {
            'total_memories': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'consolidation_runs': 0
        }

        # Consolidation patterns
        self.consolidation_patterns = self._initialize_consolidation_patterns()

        # Initialize database for persistent storage
        self._init_persistent_storage()
        
        self.is_initialized = False

    @property
    def working_memory(self):
        return self.tier_capacities.get(MemoryTier.WORKING)

    @property
    def short_term_memory(self):
        return self.tier_capacities.get(MemoryTier.SHORT_TERM)

    @property
    def long_term_memory(self):
        return self.tier_capacities.get(MemoryTier.LONG_TERM)

    @property
    def archive_memory(self):
        return self.tier_capacities.get(MemoryTier.ARCHIVE)

    @property
    def database_path(self):
        return str(self.db_path)

    async def initialize(self):
        """Start memory pyramid system"""
        self.telemetry.log_info("Starting hierarchical memory pyramid", 'hierarchical_memory_pyramid')

        # Load existing memories
        await self._load_memories_from_storage()

        # Start background processing
        self._stop_event.clear()
        self._background_thread = threading.Thread(target=self._background_processing_loop)
        self._background_thread.daemon = True
        self._background_thread.start()

        self.is_initialized = True
        self.telemetry.log_info("Hierarchical memory pyramid started", 'hierarchical_memory_pyramid')

    async def cleanup(self):
        """Stop memory pyramid system"""
        self.telemetry.log_info("Stopping hierarchical memory pyramid", 'hierarchical_memory_pyramid')

        # Stop background processing
        self._stop_event.set()
        if self._background_thread:
            self._background_thread.join(timeout=5)

        # Save all memories
        await self._save_memories_to_storage()

        self.telemetry.log_info("Hierarchical memory pyramid stopped", 'hierarchical_memory_pyramid')

    async def store_memory(self, content: Dict[str, Any], memory_type: MemoryType = MemoryType.EPISODIC,
                    priority: MemoryPriority = MemoryPriority.MEDIUM, importance: float = 0.5, 
                    tier: MemoryTier = MemoryTier.WORKING, metadata: Dict[str, Any] = None,
                    parent_memory_id: str = None) -> str:
        """Store new memory in working tier"""

        node_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        memory_node = MemoryNode(
            node_id=node_id,
            content=content,
            memory_type=memory_type,
            priority=priority,
            creation_time=now,
            last_accessed=now,
            access_count=1,
            tier=tier,
            parent_nodes=[parent_memory_id] if parent_memory_id else [],
            child_nodes=[],
            related_nodes=[],
            metadata=metadata or {},
            importance=importance
        )

        with self._lock:
            # Add to specified tier
            self.memory_tiers[tier][node_id] = memory_node

            # Update indices
            self._update_indices(memory_node)

            # Handle parent relationship
            if parent_memory_id:
                parent = await self.retrieve_memory(parent_memory_id)
                if parent:
                    if node_id not in parent.child_nodes:
                        parent.child_nodes.append(node_id)
                        # Also update parent in its tier
                        self.memory_tiers[parent.tier][parent.node_id] = parent

            # Update metrics
            self.memory_metrics['total_nodes'] += 1
            self.memory_metrics['tier_distributions'][tier.value] += 1

        # Check if tier needs consolidation
        if len(self.memory_tiers[tier]) > self.tier_capacities[tier]:
            self._trigger_consolidation(tier)

        self.telemetry.log_debug(
            f"Stored memory in {tier.value} tier: {node_id}",
            'hierarchical_memory_pyramid',
            {'memory_type': memory_type.value, 'priority': priority.value}
        )

        return node_id

    async def retrieve_memory(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory by ID from any tier"""

        with self._lock:
            # Search all tiers
            for tier in MemoryTier:
                if node_id in self.memory_tiers[tier]:
                    memory_node = self.memory_tiers[tier][node_id]

                    # Update access statistics
                    memory_node.last_accessed = datetime.now(timezone.utc)
                    memory_node.access_count += 1

                    # Consider promotion to higher tier
                    if self.auto_promotion_enabled:
                        self._consider_promotion(memory_node)

                    self.telemetry.log_debug(
                        f"Retrieved memory: {node_id} from {tier.value}",
                        'hierarchical_memory_pyramid'
                    )

                    return memory_node

        return None

    async def search_memories(self, query: Union[Dict[str, Any], str], limit: int = 10,
                       memory_type: MemoryType = None, tier: MemoryTier = None) -> List[MemoryNode]:
        """Search memories across tiers"""

        results = []

        with self._lock:
            # Determine search scope
            tiers_to_search = [tier] if tier else list(MemoryTier)

            for search_tier in tiers_to_search:
                tier_memories = self.memory_tiers[search_tier].values()

                # Filter by memory type
                if memory_type:
                    tier_memories = [m for m in tier_memories if m.memory_type == memory_type]

                # Score and rank memories
                scored_memories = []
                for memory in tier_memories:
                    if not query:
                        # Return all if query is empty (filtered by type/tier)
                        scored_memories.append((1.0, memory))
                    else:
                        score = self._calculate_relevance_score(memory, query)
                        if score > 0:
                            scored_memories.append((score, memory))

                # Sort by score
                scored_memories.sort(reverse=True)
                results.extend([memory for _, memory in scored_memories])

                if len(results) >= limit:
                    break

        # Update access statistics for retrieved memories
        for memory in results[:limit]:
            memory.last_accessed = datetime.now(timezone.utc)
            memory.access_count += 1

        self.telemetry.log_debug(
            f"Memory search returned {len(results[:limit])} results",
            'hierarchical_memory_pyramid',
            {'query': str(query), 'limit': limit}
        )

        return results[:limit]

    async def create_relationship(self, node_id1: str, node_id2: str, relationship_type: str = "related"):
        """Create relationship between two memory nodes"""

        memory1 = await self.retrieve_memory(node_id1)
        memory2 = await self.retrieve_memory(node_id2)

        if not memory1 or not memory2:
            return False

        with self._lock:
            # Add to relationship graph
            self.relationship_graph[node_id1].add(node_id2)
            self.relationship_graph[node_id2].add(node_id1)

            # Update node relationships
            if node_id2 not in memory1.related_nodes:
                memory1.related_nodes.append(node_id2)
            if node_id1 not in memory2.related_nodes:
                memory2.related_nodes.append(node_id1)

        self.telemetry.log_debug(
            f"Created relationship: {node_id1} <-> {node_id2}",
            'hierarchical_memory_pyramid',
            {'relationship_type': relationship_type}
        )

        return True

    async def get_related_memories(self, node_id: str, depth: int = 1) -> List[MemoryNode]:
        """Get related memories up to specified depth"""

        related_ids = set()
        current_level = {node_id}

        for _ in range(depth):
            next_level = set()
            for current_id in current_level:
                if current_id in self.relationship_graph:
                    next_level.update(self.relationship_graph[current_id])
            related_ids.update(next_level)
            current_level = next_level

        # Remove original node
        related_ids.discard(node_id)

        # Retrieve memory nodes
        related_memories = []
        for rel_id in related_ids:
            memory = await self.retrieve_memory(rel_id)
            if memory:
                related_memories.append(memory)

        return related_memories

    async def consolidate_memories(self, source_tier: MemoryTier, strategy: str = "auto") -> int:
        """Manually trigger memory consolidation"""

        pattern = self._select_consolidation_pattern(source_tier, strategy)
        if not pattern:
            return 0

        with self._lock:
            source_memories = list(self.memory_tiers[source_tier].values())

        # Apply consolidation pattern
        consolidated_count = self._apply_consolidation_pattern(source_memories, pattern)

        self.memory_metrics['consolidations_performed'] += 1

        self.telemetry.log_info(
            f"Consolidation completed: {consolidated_count} memories processed",
            'hierarchical_memory_pyramid',
            {
                'source_tier': source_tier.value,
                'target_tier': pattern.target_tier.value,
                'strategy': strategy
            }
        )

        return consolidated_count

    async def consolidate_by_age(self):
        """Consolidate memories based on age"""
        # Trigger consolidation for all tiers
        for tier in MemoryTier:
            await self.consolidate_memories(tier, strategy="auto")

    async def compress_memories(self, memory_ids: List[str], compression_summary: str) -> Optional[MemoryNode]:
        """Compress multiple memories into a single summary memory"""
        # Retrieve all memories
        memories = []
        for mid in memory_ids:
            mem = await self.retrieve_memory(mid)
            if mem:
                memories.append(mem)
        
        if not memories:
            return None
            
        # Create new compressed memory
        compressed_content = {
            "summary": compression_summary,
            "original_count": len(memories),
            "compressed_ids": memory_ids
        }
        
        # Store in same tier as first memory or appropriate tier
        target_tier = memories[0].tier if memories else MemoryTier.ARCHIVE
        
        new_id = await self.store_memory(
            content=compressed_content,
            memory_type=MemoryType.SEMANTIC, # Or appropriate type
            tier=target_tier,
            metadata={"is_compressed_summary": True}
        )
        
        # Mark original memories as compressed
        for memory in memories:
            memory.metadata['is_compressed'] = True
            await self.update_memory(memory)
            
        return await self.retrieve_memory(new_id)

    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""

        with self._lock:
            stats = {
                'total_memories': self.memory_metrics['total_nodes'],
                'total_nodes': self.memory_metrics['total_nodes'],
                'tier_distributions': dict(self.memory_metrics['tier_distributions']),
                'tier_capacities': {tier.value: capacity for tier, capacity in self.tier_capacities.items()},
                'tier_utilization': {},
                'memory_type_distribution': defaultdict(int),
                'priority_distribution': defaultdict(int),
                'access_patterns': {},
                'consolidation_metrics': {
                    'consolidations_performed': self.memory_metrics['consolidations_performed'],
                    'promotions': self.memory_metrics['promotions'],
                    'demotions': self.memory_metrics['demotions'],
                    'compressions': self.memory_metrics['compressions']
                }
            }

            # Calculate tier utilization
            for tier in MemoryTier:
                current_count = len(self.memory_tiers[tier])
                capacity = self.tier_capacities[tier]
                stats['tier_utilization'][tier.value] = (current_count / capacity) * 100

            # Analyze memory distributions
            for tier_memories in self.memory_tiers.values():
                for memory in tier_memories.values():
                    stats['memory_type_distribution'][memory.memory_type.value] += 1
                    stats['priority_distribution'][memory.priority.value] += 1

            # Calculate access patterns
            recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_accesses = 0
            total_accesses = 0

            for tier_memories in self.memory_tiers.values():
                for memory in tier_memories.values():
                    total_accesses += memory.access_count
                    if memory.last_accessed > recent_cutoff:
                        recent_accesses += 1

            stats['access_patterns'] = {
                'total_accesses': total_accesses,
                'recent_accesses_24h': recent_accesses,
                'avg_accesses_per_node': total_accesses / max(stats['total_nodes'], 1)
            }

        return stats

    async def get_memories_by_tier(self, tier: MemoryTier) -> List[MemoryNode]:
        """Get all memories in a specific tier"""
        with self._lock:
            return list(self.memory_tiers[tier].values())

    async def get_memories_by_importance(self, min_importance: float) -> List[MemoryNode]:
        """Get memories with importance score above threshold"""
        results = []
        with self._lock:
            for tier in MemoryTier:
                for memory in self.memory_tiers[tier].values():
                    if memory.importance >= min_importance:
                        results.append(memory)
        return sorted(results, key=lambda m: m.importance, reverse=True)

    async def get_most_accessed_memories(self, limit: int = 10) -> List[MemoryNode]:
        """Get most frequently accessed memories"""
        results = []
        with self._lock:
            for tier in MemoryTier:
                results.extend(self.memory_tiers[tier].values())
        return sorted(results, key=lambda m: m.access_count, reverse=True)[:limit]

    async def get_child_memories(self, parent_id: str) -> List[MemoryNode]:
        """Get child memories for a parent node"""
        # This implementation assumes child_nodes attribute is populated or we search
        # Since we don't have a direct index, we search all (inefficient but works for now)
        # Or better, we rely on parent_nodes in children if we had an index.
        # But MemoryNode has child_nodes list.
        parent = await self.retrieve_memory(parent_id)
        if not parent:
            return []
        
        children = []
        for child_id in parent.child_nodes:
            child = await self.retrieve_memory(child_id)
            if child:
                children.append(child)
        return children

    async def search_memories_by_type(self, memory_type: MemoryType) -> List[MemoryNode]:
        """Search memories by type"""
        return await self.search_memories(query={}, memory_type=memory_type)

    async def cleanup_low_importance(self):
        """Cleanup low importance memories from archive"""
        # Simplified implementation
        with self._lock:
            archive = self.memory_tiers[MemoryTier.ARCHIVE]
            to_remove = []
            for node_id, memory in archive.items():
                if memory.importance < 0.2:
                    to_remove.append(node_id)
            
            for node_id in to_remove:
                del archive[node_id]

    async def export_memories(self, path: Path):
        """Export memories to JSON file"""
        export_data = []
        with self._lock:
            for tier in MemoryTier:
                for memory in self.memory_tiers[tier].values():
                    export_data.append(memory.to_dict())
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

    async def import_memories(self, path: Path):
        """Import memories from JSON file"""
        if not path.exists():
            return
            
        with open(path, 'r') as f:
            import_data = json.load(f)
            
        for data in import_data:
            memory = MemoryNode.from_dict(data)
            with self._lock:
                self.memory_tiers[memory.tier][memory.node_id] = memory
                self._update_indices(memory)
                
        return len(import_data)

    async def update_memory(self, memory: MemoryNode):
        """Update an existing memory node"""
        with self._lock:
            if memory.node_id in self.memory_tiers[memory.tier]:
                self.memory_tiers[memory.tier][memory.node_id] = memory
                # Update indices if needed (simplified)

    # Internal methods

    def _initialize_consolidation_patterns(self) -> List[ConsolidationPattern]:
        """Initialize consolidation patterns for different scenarios"""

        patterns = [
            ConsolidationPattern(
                pattern_id="working_to_short_term",
                source_tier=MemoryTier.WORKING,
                target_tier=MemoryTier.SHORT_TERM,
                conditions={
                    'age_hours': 1,
                    'access_threshold': 2,
                    'priority_min': MemoryPriority.MEDIUM
                },
                strategy="frequency_based",
                compression_algorithm="none",
                retention_policy={'keep_percentage': 80}
            ),
            ConsolidationPattern(
                pattern_id="short_term_to_long_term",
                source_tier=MemoryTier.SHORT_TERM,
                target_tier=MemoryTier.LONG_TERM,
                conditions={
                    'age_days': 7,
                    'access_threshold': 3,
                    'priority_min': MemoryPriority.MEDIUM
                },
                strategy="semantic_clustering",
                compression_algorithm="lightweight",
                retention_policy={'keep_percentage': 60}
            ),
            ConsolidationPattern(
                pattern_id="long_term_to_archive",
                source_tier=MemoryTier.LONG_TERM,
                target_tier=MemoryTier.ARCHIVE,
                conditions={
                    'age_months': 6,
                    'access_threshold': 1,
                    'priority_min': MemoryPriority.LOW
                },
                strategy="compression_focused",
                compression_algorithm="aggressive",
                retention_policy={'keep_percentage': 40}
            )
        ]

        return patterns

    def _update_indices(self, memory_node: MemoryNode):
        """Update search indices with new memory node"""

        # Type index
        self.type_index[memory_node.memory_type].add(memory_node.node_id)

        # Priority index
        self.priority_index[memory_node.priority].add(memory_node.node_id)

        # Temporal index
        date_key = memory_node.creation_time.date().isoformat()
        self.temporal_index[date_key].append(memory_node.node_id)

    def _calculate_relevance_score(self, memory: MemoryNode, query: Dict[str, Any]) -> float:
        """Calculate relevance score for search query"""

        score = 0.0

        # Content similarity (simplified)
        if isinstance(query, str):
            # String query
            if isinstance(memory.content, str):
                if query.lower() in memory.content.lower():
                    score += 1.0
            elif isinstance(memory.content, dict):
                for key, value in memory.content.items():
                    if isinstance(value, str) and query.lower() in value.lower():
                        score += 1.0
        else:
            # Dict query
            if isinstance(memory.content, dict):
                for key, value in query.items():
                    if key in memory.content:
                        if memory.content[key] == value:
                            score += 1.0
                        elif isinstance(value, str) and isinstance(memory.content[key], str):
                            # Simple substring matching
                            if value.lower() in memory.content[key].lower():
                                score += 0.5

        # Metadata matching
        if isinstance(query, dict):
            for key, value in query.items():
                if key in memory.metadata and memory.metadata[key] == value:
                    score += 0.5

        # Boost score based on priority
        priority_boost = {
            MemoryPriority.CRITICAL: 2.0,
            MemoryPriority.HIGH: 1.5,
            MemoryPriority.MEDIUM: 1.0,
            MemoryPriority.LOW: 0.5
        }
        score *= priority_boost.get(memory.priority, 1.0)

        # Boost recent memories
        age_hours = (datetime.now(timezone.utc) - memory.last_accessed).total_seconds() / 3600
        if age_hours < 24:
            score *= 1.2
        elif age_hours < 168:  # 1 week
            score *= 1.1

        return score

    def _trigger_consolidation(self, tier: MemoryTier):
        """Trigger consolidation for specified tier"""

        if not self.consolidation_enabled:
            return

        # Find appropriate consolidation pattern
        pattern = self._select_consolidation_pattern(tier)
        if pattern:
            # Schedule consolidation in background
            self._schedule_consolidation(pattern)

    def _select_consolidation_pattern(self, source_tier: MemoryTier, strategy: str = "auto") -> Optional[ConsolidationPattern]:
        """Select appropriate consolidation pattern"""

        for pattern in self.consolidation_patterns:
            if pattern.source_tier == source_tier:
                if strategy == "auto" or pattern.strategy == strategy:
                    return pattern

        return None

    def _apply_consolidation_pattern(self, memories: List[MemoryNode], pattern: ConsolidationPattern) -> int:
        """Apply consolidation pattern to memories"""

        # Filter memories based on pattern conditions
        eligible_memories = self._filter_memories_by_conditions(memories, pattern.conditions)

        # Apply retention policy
        retained_count = math.ceil(len(eligible_memories) * pattern.retention_policy['keep_percentage'] / 100)
        memories_to_consolidate = self._select_memories_for_consolidation(eligible_memories, retained_count, pattern.strategy)

        # Move memories to target tier
        consolidated_count = 0
        for memory in memories_to_consolidate:
            if self._move_memory_to_tier(memory, pattern.target_tier):

                # Apply compression if specified
                if pattern.compression_algorithm != "none":
                    self._compress_memory(memory, pattern.compression_algorithm)

                consolidated_count += 1

        return consolidated_count

    def _filter_memories_by_conditions(self, memories: List[MemoryNode], conditions: Dict[str, Any]) -> List[MemoryNode]:
        """Filter memories based on consolidation conditions"""

        now = datetime.now(timezone.utc)
        eligible = []

        for memory in memories:
            # Age condition
            age = now - memory.creation_time

            if 'age_hours' in conditions:
                if age.total_seconds() / 3600 < conditions['age_hours']:
                    continue

            if 'age_days' in conditions:
                if age.days < conditions['age_days']:
                    continue

            if 'age_months' in conditions:
                if age.days < conditions['age_months'] * 30:
                    continue

            # Access threshold
            if 'access_threshold' in conditions:
                if memory.access_count < conditions['access_threshold']:
                    continue

            # Priority minimum
            if 'priority_min' in conditions:
                min_priority = conditions['priority_min']
                priority_order = [MemoryPriority.LOW, MemoryPriority.MEDIUM, MemoryPriority.HIGH, MemoryPriority.CRITICAL]
                if priority_order.index(memory.priority) < priority_order.index(min_priority):
                    continue

            eligible.append(memory)

        return eligible

    def _select_memories_for_consolidation(self, memories: List[MemoryNode], target_count: int, strategy: str) -> List[MemoryNode]:
        """Select specific memories for consolidation based on strategy"""

        if strategy == "frequency_based":
            # Sort by access count and last accessed time
            memories.sort(key=lambda m: (m.access_count, m.last_accessed), reverse=True)

        elif strategy == "semantic_clustering":
            # Group similar memories (simplified implementation)
            memories.sort(key=lambda m: len(m.related_nodes), reverse=True)

        elif strategy == "compression_focused":
            # Prioritize older, less accessed memories
            memories.sort(key=lambda m: (m.last_accessed, -m.access_count))

        return memories[:target_count]

    def _move_memory_to_tier(self, memory: MemoryNode, target_tier: MemoryTier) -> bool:
        """Move memory from current tier to target tier"""

        with self._lock:
            # Check target tier capacity
            if len(self.memory_tiers[target_tier]) >= self.tier_capacities[target_tier]:
                return False

            # Remove from current tier
            current_tier = memory.tier
            if memory.node_id in self.memory_tiers[current_tier]:
                del self.memory_tiers[current_tier][memory.node_id]
                self.memory_metrics['tier_distributions'][current_tier.value] -= 1

            # Add to target tier
            memory.tier = target_tier
            self.memory_tiers[target_tier][memory.node_id] = memory
            self.memory_metrics['tier_distributions'][target_tier.value] += 1

        return True

    def _compress_memory(self, memory: MemoryNode, algorithm: str):
        """Apply compression to memory content"""

        if algorithm == "lightweight":
            # Remove non-essential metadata
            if 'debug_info' in memory.metadata:
                del memory.metadata['debug_info']
            memory.compression_ratio = 0.8

        elif algorithm == "aggressive":
            # Compress content and metadata significantly
            # Keep only essential information
            essential_keys = ['id', 'type', 'summary', 'outcome']
            compressed_content = {k: v for k, v in memory.content.items() if k in essential_keys}
            memory.content = compressed_content
            memory.metadata = {'compressed': True, 'algorithm': algorithm}
            memory.compression_ratio = 0.3

        self.memory_metrics['compressions'] += 1

    def _consider_promotion(self, memory: MemoryNode):
        """Consider promoting memory to higher tier based on access patterns"""

        if not self.auto_promotion_enabled:
            return

        current_tier = memory.tier

        # Promotion criteria
        should_promote = False

        if current_tier == MemoryTier.WORKING:
            # Promote to short-term if frequently accessed
            if memory.access_count >= 5:
                should_promote = True
                target_tier = MemoryTier.SHORT_TERM

        elif current_tier == MemoryTier.SHORT_TERM:
            # Promote to long-term if consistently accessed over time
            age_days = (datetime.now(timezone.utc) - memory.creation_time).days
            if age_days >= 7 and memory.access_count >= 10:
                should_promote = True
                target_tier = MemoryTier.LONG_TERM
            # Recall to working memory if very hot
            elif memory.access_count >= 10:
                should_promote = True
                target_tier = MemoryTier.WORKING

        if should_promote:
            if self._move_memory_to_tier(memory, target_tier):
                self.memory_metrics['promotions'] += 1
                self.telemetry.log_debug(
                    f"Promoted memory {memory.node_id} to {target_tier.value}",
                    'hierarchical_memory_pyramid'
                )

    def _schedule_consolidation(self, pattern: ConsolidationPattern):
        """Schedule consolidation to run in background"""

        # Simple implementation - in production would use a proper task queue
        def run_consolidation():
            time.sleep(1)  # Brief delay
            with self._lock:
                memories = list(self.memory_tiers[pattern.source_tier].values())
            self._apply_consolidation_pattern(memories, pattern)

        consolidation_thread = threading.Thread(target=run_consolidation)
        consolidation_thread.daemon = True
        consolidation_thread.start()

    def _background_processing_loop(self):
        """Background processing loop for maintenance tasks"""

        while not self._stop_event.is_set():
            try:
                # Check for consolidation needs
                self._check_consolidation_needs()

                # Update memory access statistics
                self._update_memory_statistics()

                # Periodic cleanup
                self._cleanup_expired_indices()

                # Sleep between processing cycles (interruptible)
                if self._stop_event.wait(timeout=1): # Reduced check interval
                    break

            except Exception as e:
                self.telemetry.log_error(
                    f"Error in memory pyramid background processing: {e}",
                    'hierarchical_memory_pyramid'
                )
                if self._stop_event.wait(timeout=5):
                    break

    def _check_consolidation_needs(self):
        """Check if any tier needs consolidation"""

        for tier in MemoryTier:
            current_count = len(self.memory_tiers[tier])
            capacity = self.tier_capacities[tier]

            # Trigger consolidation if tier is over 90% capacity
            if current_count / capacity > 0.9:
                self._trigger_consolidation(tier)

    def _update_memory_statistics(self):
        """Update memory system statistics"""

        total_nodes = sum(len(tier_memories) for tier_memories in self.memory_tiers.values())
        self.memory_metrics['total_nodes'] = total_nodes

        for tier in MemoryTier:
            self.memory_metrics['tier_distributions'][tier.value] = len(self.memory_tiers[tier])

    def _cleanup_expired_indices(self):
        """Clean up expired entries from temporal indices"""

        # Remove entries older than 1 year from temporal index
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=365)).date().isoformat()

        with self._lock:
            expired_dates = [date for date in self.temporal_index.keys() if date < cutoff_date]
            for date in expired_dates:
                del self.temporal_index[date]

    def _init_persistent_storage(self):
        """Initialize persistent storage for memories"""

        if self.db_path_override:
            self.db_path = self.db_path_override
        else:
            self.db_path = self.storage_path / "memory_pyramid.db"

        # Create database schema
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    node_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    creation_time TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    parent_nodes TEXT DEFAULT '',
                    child_nodes TEXT DEFAULT '',
                    related_nodes TEXT DEFAULT '',
                    embedding BLOB,
                    compression_ratio REAL DEFAULT 1.0,
                    importance REAL DEFAULT 0.5
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tier ON memory_nodes(tier)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON memory_nodes(memory_type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_priority ON memory_nodes(priority)
            """)

    async def _save_memories_to_storage(self):
        """Save memories to persistent storage"""

        self.telemetry.log_info("Saving memories to persistent storage", 'hierarchical_memory_pyramid')

        with sqlite3.connect(str(self.db_path)) as conn:
            # Clear existing data
            conn.execute("DELETE FROM memory_nodes")

            # Save all memories
            for tier_memories in self.memory_tiers.values():
                for memory in tier_memories.values():
                    conn.execute("""
                        INSERT INTO memory_nodes (
                            node_id, tier, memory_type, priority, content, metadata,
                            creation_time, last_accessed, access_count,
                            parent_nodes, child_nodes, related_nodes,
                            embedding, compression_ratio, importance
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        memory.node_id,
                        memory.tier.value,
                        memory.memory_type.value,
                        memory.priority.value,
                        json.dumps(memory.content),
                        json.dumps(memory.metadata),
                        memory.creation_time.isoformat(),
                        memory.last_accessed.isoformat(),
                        memory.access_count,
                        json.dumps(memory.parent_nodes),
                        json.dumps(memory.child_nodes),
                        json.dumps(memory.related_nodes),
                        pickle.dumps(memory.embedding) if memory.embedding else None,
                        memory.compression_ratio,
                        memory.importance
                    ))

        self.telemetry.log_info(
            f"Saved {self.memory_metrics['total_nodes']} memories to persistent storage",
            'hierarchical_memory_pyramid'
        )

    async def _load_memories_from_storage(self):
        """Load memories from persistent storage"""

        if not self.db_path.exists():
            return

        self.telemetry.log_info("Loading memories from persistent storage", 'hierarchical_memory_pyramid')

        loaded_count = 0

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT node_id, tier, memory_type, priority, content, metadata,
                       creation_time, last_accessed, access_count,
                       parent_nodes, child_nodes, related_nodes,
                       embedding, compression_ratio, importance
                FROM memory_nodes
            """)

            for row in cursor:
                try:
                    memory = MemoryNode(
                        node_id=row[0],
                        tier=MemoryTier(row[1]),
                        memory_type=MemoryType(row[2]),
                        priority=MemoryPriority(row[3]),
                        content=json.loads(row[4]),
                        metadata=json.loads(row[5]),
                        creation_time=datetime.fromisoformat(row[6]),
                        last_accessed=datetime.fromisoformat(row[7]),
                        access_count=row[8],
                        parent_nodes=json.loads(row[9]) if row[9] else [],
                        child_nodes=json.loads(row[10]) if row[10] else [],
                        related_nodes=json.loads(row[11]) if row[11] else [],
                        embedding=pickle.loads(row[12]) if row[12] else None,
                        compression_ratio=row[13] if row[13] else 1.0,
                        importance=row[14] if len(row) > 14 else 0.5
                    )

                    # Add to appropriate tier
                    self.memory_tiers[memory.tier][memory.node_id] = memory

                    # Update indices
                    self._update_indices(memory)

                    # Update relationship graph
                    for related_id in memory.related_nodes:
                        self.relationship_graph[memory.node_id].add(related_id)

                    loaded_count += 1

                except Exception as e:
                    self.telemetry.log_warning(
                        f"Failed to load memory node: {e}",
                        'hierarchical_memory_pyramid'
                    )

        # Update metrics
        self._update_memory_statistics()

        self.telemetry.log_info(
            f"Loaded {loaded_count} memories from persistent storage",
            'hierarchical_memory_pyramid'
        )