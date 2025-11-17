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
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
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

class MemoryPriority(Enum):
    CRITICAL = "critical"      # Must retain
    HIGH = "high"             # Important to retain
    MEDIUM = "medium"         # Moderately important
    LOW = "low"              # Can be discarded if needed

@dataclass
class MemoryNode:
    """Individual memory unit in the hierarchy"""
    node_id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    priority: MemoryPriority
    creation_time: datetime
    last_accessed: datetime
    access_count: int
    tier: MemoryTier

    # Relationships
    parent_nodes: List[str]
    child_nodes: List[str]
    related_nodes: List[str]

    # Metadata
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    compression_ratio: float = 1.0

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
            'compression_ratio': self.compression_ratio
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
            compression_ratio=data.get('compression_ratio', 1.0)
        )

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

    def __init__(self, storage_path: str = "./memory_pyramid", telemetry=None):
        self.telemetry = telemetry or get_telemetry()
        self.telemetry.register_component('hierarchical_memory_pyramid')

        # Storage configuration
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

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
        self._lock = threading.Lock()
        self._background_active = False
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

        # Consolidation patterns
        self.consolidation_patterns = self._initialize_consolidation_patterns()

        # Initialize database for persistent storage
        self._init_persistent_storage()

    async def start(self):
        """Start memory pyramid system"""
        self.telemetry.log_info("Starting hierarchical memory pyramid", 'hierarchical_memory_pyramid')

        # Load existing memories
        await self._load_memories_from_storage()

        # Start background processing
        self._background_active = True
        self._background_thread = threading.Thread(target=self._background_processing_loop)
        self._background_thread.daemon = True
        self._background_thread.start()

        self.telemetry.log_info("Hierarchical memory pyramid started", 'hierarchical_memory_pyramid')

    async def stop(self):
        """Stop memory pyramid system"""
        self.telemetry.log_info("Stopping hierarchical memory pyramid", 'hierarchical_memory_pyramid')

        # Stop background processing
        self._background_active = False
        if self._background_thread:
            self._background_thread.join(timeout=10)

        # Save all memories
        await self._save_memories_to_storage()

        self.telemetry.log_info("Hierarchical memory pyramid stopped", 'hierarchical_memory_pyramid')

    def store_memory(self, content: Dict[str, Any], memory_type: MemoryType = MemoryType.EPISODIC,
                    priority: MemoryPriority = MemoryPriority.MEDIUM, metadata: Dict[str, Any] = None) -> str:
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
            tier=MemoryTier.WORKING,
            parent_nodes=[],
            child_nodes=[],
            related_nodes=[],
            metadata=metadata or {}
        )

        with self._lock:
            # Add to working memory
            self.memory_tiers[MemoryTier.WORKING][node_id] = memory_node

            # Update indices
            self._update_indices(memory_node)

            # Update metrics
            self.memory_metrics['total_nodes'] += 1
            self.memory_metrics['tier_distributions'][MemoryTier.WORKING.value] += 1

        # Check if working memory needs consolidation
        if len(self.memory_tiers[MemoryTier.WORKING]) > self.tier_capacities[MemoryTier.WORKING]:
            self._trigger_consolidation(MemoryTier.WORKING)

        self.telemetry.log_debug(
            f"Stored memory in working tier: {node_id}",
            'hierarchical_memory_pyramid',
            {'memory_type': memory_type.value, 'priority': priority.value}
        )

        return node_id

    def retrieve_memory(self, node_id: str) -> Optional[MemoryNode]:
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

    def search_memories(self, query: Dict[str, Any], limit: int = 10,
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
            {'query_keys': list(query.keys()), 'limit': limit}
        )

        return results[:limit]

    def create_relationship(self, node_id1: str, node_id2: str, relationship_type: str = "related"):
        """Create relationship between two memory nodes"""

        memory1 = self.retrieve_memory(node_id1)
        memory2 = self.retrieve_memory(node_id2)

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

    def get_related_memories(self, node_id: str, depth: int = 1) -> List[MemoryNode]:
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
            memory = self.retrieve_memory(rel_id)
            if memory:
                related_memories.append(memory)

        return related_memories

    def consolidate_memories(self, source_tier: MemoryTier, strategy: str = "auto") -> int:
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

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""

        with self._lock:
            stats = {
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
        for key, value in query.items():
            if key in memory.content:
                if memory.content[key] == value:
                    score += 1.0
                elif isinstance(value, str) and isinstance(memory.content[key], str):
                    # Simple substring matching
                    if value.lower() in memory.content[key].lower():
                        score += 0.5

        # Metadata matching
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
        retained_count = int(len(eligible_memories) * pattern.retention_policy['keep_percentage'] / 100)
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

        while self._background_active:
            try:
                # Check for consolidation needs
                self._check_consolidation_needs()

                # Update memory access statistics
                self._update_memory_statistics()

                # Periodic cleanup
                self._cleanup_expired_indices()

                # Sleep between processing cycles
                time.sleep(30)  # 30 seconds

            except Exception as e:
                self.telemetry.log_error(
                    f"Error in memory pyramid background processing: {e}",
                    'hierarchical_memory_pyramid'
                )
                time.sleep(30)

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
                    compression_ratio REAL DEFAULT 1.0
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
                            embedding, compression_ratio
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        memory.compression_ratio
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
                       embedding, compression_ratio
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
                        compression_ratio=row[13] if row[13] else 1.0
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