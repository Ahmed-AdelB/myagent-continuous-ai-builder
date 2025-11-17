"""
Unit tests for GPT-5 Priority 4: Hierarchical Memory Pyramid
Tests the 4-tier memory architecture with automatic consolidation
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from core.memory_pyramid.hierarchical_memory_pyramid import (
    HierarchicalMemoryPyramid,
    MemoryNode,
    MemoryTier,
    MemoryType
)


@pytest.mark.unit
@pytest.mark.gpt5
@pytest.mark.memory
class TestHierarchicalMemoryPyramid:
    """Test suite for Hierarchical Memory Pyramid"""

    @pytest.fixture
    async def memory_pyramid(self, temp_database):
        """Create memory pyramid instance for testing"""
        pyramid = HierarchicalMemoryPyramid(database_path=temp_database)
        await pyramid.initialize()
        yield pyramid
        await pyramid.cleanup()

    @pytest.mark.asyncio
    async def test_initialization(self, temp_database):
        """Test memory pyramid initialization"""
        pyramid = HierarchicalMemoryPyramid(database_path=temp_database)

        # Test initialization
        await pyramid.initialize()

        # Verify tiers are created
        assert pyramid.working_memory is not None
        assert pyramid.short_term_memory is not None
        assert pyramid.long_term_memory is not None
        assert pyramid.archive_memory is not None

        # Verify database setup
        assert pyramid.database_path == temp_database
        assert pyramid.is_initialized

        await pyramid.cleanup()

    @pytest.mark.asyncio
    async def test_store_memory_working_tier(self, memory_pyramid):
        """Test storing memory in working tier"""
        content = "User requested fibonacci function implementation"
        memory_type = MemoryType.USER_REQUEST

        memory_id = await memory_pyramid.store_memory(
            content=content,
            memory_type=memory_type,
            importance=0.8
        )

        assert memory_id is not None

        # Verify memory is in working tier
        working_memories = await memory_pyramid.get_memories_by_tier(MemoryTier.WORKING)
        assert len(working_memories) == 1
        assert working_memories[0].content == content
        assert working_memories[0].memory_type == memory_type
        assert working_memories[0].importance == 0.8

    @pytest.mark.asyncio
    async def test_store_multiple_memories(self, memory_pyramid):
        """Test storing multiple memories in different tiers"""
        memories = [
            {
                "content": "User request for calculator",
                "memory_type": MemoryType.USER_REQUEST,
                "importance": 0.7,
                "tier": MemoryTier.WORKING
            },
            {
                "content": "Implemented basic calculator class",
                "memory_type": MemoryType.IMPLEMENTATION,
                "importance": 0.8,
                "tier": MemoryTier.SHORT_TERM
            },
            {
                "content": "Performance optimization applied",
                "memory_type": MemoryType.OPTIMIZATION,
                "importance": 0.9,
                "tier": MemoryTier.LONG_TERM
            }
        ]

        memory_ids = []
        for memory_data in memories:
            memory_id = await memory_pyramid.store_memory(
                content=memory_data["content"],
                memory_type=memory_data["memory_type"],
                importance=memory_data["importance"],
                tier=memory_data["tier"]
            )
            memory_ids.append(memory_id)

        # Verify all memories stored
        assert len(memory_ids) == 3
        assert all(mid is not None for mid in memory_ids)

        # Verify memories in correct tiers
        working_memories = await memory_pyramid.get_memories_by_tier(MemoryTier.WORKING)
        short_term_memories = await memory_pyramid.get_memories_by_tier(MemoryTier.SHORT_TERM)
        long_term_memories = await memory_pyramid.get_memories_by_tier(MemoryTier.LONG_TERM)

        assert len(working_memories) == 1
        assert len(short_term_memories) == 1
        assert len(long_term_memories) == 1

    @pytest.mark.asyncio
    async def test_retrieve_memory_by_id(self, memory_pyramid):
        """Test retrieving specific memory by ID"""
        content = "Test memory for retrieval"
        memory_type = MemoryType.DECISION

        memory_id = await memory_pyramid.store_memory(
            content=content,
            memory_type=memory_type,
            importance=0.6
        )

        # Retrieve memory
        retrieved_memory = await memory_pyramid.retrieve_memory(memory_id)

        assert retrieved_memory is not None
        assert retrieved_memory.memory_id == memory_id
        assert retrieved_memory.content == content
        assert retrieved_memory.memory_type == memory_type
        assert retrieved_memory.importance == 0.6

    @pytest.mark.asyncio
    async def test_search_memories_by_content(self, memory_pyramid):
        """Test searching memories by content"""
        # Store test memories
        test_memories = [
            "Implement fibonacci function using recursion",
            "Optimize fibonacci with memoization",
            "Create unit tests for calculator module",
            "Debug performance issues in sorting algorithm"
        ]

        for content in test_memories:
            await memory_pyramid.store_memory(
                content=content,
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.7
            )

        # Search for fibonacci-related memories
        fibonacci_memories = await memory_pyramid.search_memories("fibonacci")
        assert len(fibonacci_memories) == 2

        # Search for testing-related memories
        test_related = await memory_pyramid.search_memories("test")
        assert len(test_related) == 1

        # Search for non-existent content
        no_results = await memory_pyramid.search_memories("nonexistent")
        assert len(no_results) == 0

    @pytest.mark.asyncio
    async def test_search_memories_by_type(self, memory_pyramid):
        """Test searching memories by type"""
        # Store memories of different types
        memories_by_type = [
            ("User wants calculator", MemoryType.USER_REQUEST),
            ("Implemented Calculator class", MemoryType.IMPLEMENTATION),
            ("Decided to use Python", MemoryType.DECISION),
            ("Fixed division by zero bug", MemoryType.BUG_FIX),
            ("Optimized algorithm performance", MemoryType.OPTIMIZATION)
        ]

        for content, mem_type in memories_by_type:
            await memory_pyramid.store_memory(
                content=content,
                memory_type=mem_type,
                importance=0.7
            )

        # Search by specific types
        implementations = await memory_pyramid.search_memories_by_type(MemoryType.IMPLEMENTATION)
        decisions = await memory_pyramid.search_memories_by_type(MemoryType.DECISION)
        bug_fixes = await memory_pyramid.search_memories_by_type(MemoryType.BUG_FIX)

        assert len(implementations) == 1
        assert len(decisions) == 1
        assert len(bug_fixes) == 1

    @pytest.mark.asyncio
    async def test_memory_consolidation_by_count(self, memory_pyramid):
        """Test automatic memory consolidation when working memory is full"""
        # Set low threshold for testing
        pyramid = memory_pyramid
        pyramid.working_memory_threshold = 3

        # Fill working memory beyond threshold
        for i in range(5):
            await pyramid.store_memory(
                content=f"Working memory item {i}",
                memory_type=MemoryType.USER_REQUEST,
                importance=0.5 + (i * 0.1)  # Varying importance
            )

        # Trigger consolidation
        await pyramid.consolidate_memories()

        # Verify consolidation occurred
        working_memories = await pyramid.get_memories_by_tier(MemoryTier.WORKING)
        short_term_memories = await pyramid.get_memories_by_tier(MemoryTier.SHORT_TERM)

        # High importance memories should remain in working
        # Lower importance should move to short-term
        assert len(working_memories) <= pyramid.working_memory_threshold
        assert len(short_term_memories) > 0

    @pytest.mark.asyncio
    async def test_memory_consolidation_by_time(self, memory_pyramid):
        """Test time-based memory consolidation"""
        pyramid = memory_pyramid

        # Create old memory (simulate)
        old_content = "Old memory for consolidation test"
        memory_id = await pyramid.store_memory(
            content=old_content,
            memory_type=MemoryType.IMPLEMENTATION,
            importance=0.6
        )

        # Manually update timestamp to simulate age
        with patch('datetime.datetime') as mock_datetime:
            # Mock old timestamp
            old_time = datetime.now() - timedelta(hours=25)  # Older than 24 hours
            mock_datetime.now.return_value = old_time

            # Trigger time-based consolidation
            await pyramid.consolidate_by_age()

        # Verify memory moved to appropriate tier
        working_memories = await pyramid.get_memories_by_tier(MemoryTier.WORKING)
        short_term_memories = await pyramid.get_memories_by_tier(MemoryTier.SHORT_TERM)

        # Memory should have moved from working to short-term
        working_contents = [m.content for m in working_memories]
        short_term_contents = [m.content for m in short_term_memories]

        assert old_content not in working_contents or old_content in short_term_contents

    @pytest.mark.asyncio
    async def test_memory_compression(self, memory_pyramid):
        """Test memory compression for archive tier"""
        pyramid = memory_pyramid

        # Create memories for compression
        related_memories = [
            "Implemented fibonacci function",
            "Added memoization to fibonacci",
            "Optimized fibonacci performance",
            "Fixed fibonacci edge cases"
        ]

        memory_ids = []
        for content in related_memories:
            memory_id = await pyramid.store_memory(
                content=content,
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.7,
                tier=MemoryTier.LONG_TERM
            )
            memory_ids.append(memory_id)

        # Compress related memories
        compressed_memory = await pyramid.compress_memories(
            memory_ids=memory_ids,
            compression_summary="Complete fibonacci implementation with optimizations"
        )

        assert compressed_memory is not None
        assert "fibonacci" in compressed_memory.content
        assert compressed_memory.tier == MemoryTier.ARCHIVE

        # Original memories should be marked as compressed
        for memory_id in memory_ids:
            memory = await pyramid.retrieve_memory(memory_id)
            assert memory.compressed is True

    @pytest.mark.asyncio
    async def test_memory_importance_scoring(self, memory_pyramid):
        """Test memory importance calculation and ranking"""
        # Store memories with different characteristics
        memories = [
            {
                "content": "Critical security vulnerability found",
                "memory_type": MemoryType.BUG_FIX,
                "importance": 0.95,
                "access_count": 10
            },
            {
                "content": "Minor UI improvement suggestion",
                "memory_type": MemoryType.IMPROVEMENT,
                "importance": 0.3,
                "access_count": 1
            },
            {
                "content": "Core algorithm implementation",
                "memory_type": MemoryType.IMPLEMENTATION,
                "importance": 0.8,
                "access_count": 5
            }
        ]

        memory_ids = []
        for memory_data in memories:
            memory_id = await memory_pyramid.store_memory(
                content=memory_data["content"],
                memory_type=memory_data["memory_type"],
                importance=memory_data["importance"]
            )
            memory_ids.append(memory_id)

            # Simulate access count
            memory = await memory_pyramid.retrieve_memory(memory_id)
            memory.access_count = memory_data["access_count"]
            await memory_pyramid.update_memory(memory)

        # Get memories by importance
        important_memories = await memory_pyramid.get_memories_by_importance(min_importance=0.7)
        assert len(important_memories) == 2  # Security bug and core algorithm

        # Get most accessed memories
        accessed_memories = await memory_pyramid.get_most_accessed_memories(limit=2)
        assert len(accessed_memories) == 2
        assert accessed_memories[0].access_count >= accessed_memories[1].access_count

    @pytest.mark.asyncio
    async def test_memory_relationships(self, memory_pyramid):
        """Test memory relationship tracking"""
        # Create parent memory
        parent_id = await memory_pyramid.store_memory(
            content="User request for calculator application",
            memory_type=MemoryType.USER_REQUEST,
            importance=0.8
        )

        # Create related memories
        child_memories = [
            "Implemented Calculator class",
            "Added unit tests for Calculator",
            "Optimized calculation performance"
        ]

        child_ids = []
        for content in child_memories:
            child_id = await memory_pyramid.store_memory(
                content=content,
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.7,
                parent_memory_id=parent_id
            )
            child_ids.append(child_id)

        # Verify relationships
        children = await memory_pyramid.get_child_memories(parent_id)
        assert len(children) == 3

        # Verify parent relationship
        for child_id in child_ids:
            child_memory = await memory_pyramid.retrieve_memory(child_id)
            assert child_memory.parent_memory_id == parent_id

    @pytest.mark.asyncio
    async def test_memory_statistics(self, memory_pyramid):
        """Test memory usage statistics"""
        # Add various memories
        for i in range(10):
            await memory_pyramid.store_memory(
                content=f"Test memory {i}",
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.5 + (i * 0.05),
                tier=MemoryTier.WORKING if i < 3 else MemoryTier.SHORT_TERM
            )

        # Get statistics
        stats = await memory_pyramid.get_memory_statistics()

        assert stats["total_memories"] == 10
        assert stats["working_memory_count"] == 3
        assert stats["short_term_memory_count"] == 7
        assert "average_importance" in stats
        assert "tier_distribution" in stats
        assert "memory_types_distribution" in stats

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_pyramid):
        """Test memory cleanup and garbage collection"""
        # Create low-importance memories
        low_importance_ids = []
        for i in range(5):
            memory_id = await memory_pyramid.store_memory(
                content=f"Low importance memory {i}",
                memory_type=MemoryType.OPTIMIZATION,
                importance=0.1,
                tier=MemoryTier.ARCHIVE
            )
            low_importance_ids.append(memory_id)

        # Create high-importance memory
        high_importance_id = await memory_pyramid.store_memory(
            content="Critical system information",
            memory_type=MemoryType.DECISION,
            importance=0.95,
            tier=MemoryTier.LONG_TERM
        )

        # Perform cleanup (remove memories below threshold)
        deleted_count = await memory_pyramid.cleanup_low_importance_memories(threshold=0.2)

        assert deleted_count == 5

        # Verify high-importance memory remains
        high_importance_memory = await memory_pyramid.retrieve_memory(high_importance_id)
        assert high_importance_memory is not None

        # Verify low-importance memories removed
        for memory_id in low_importance_ids:
            deleted_memory = await memory_pyramid.retrieve_memory(memory_id)
            assert deleted_memory is None or deleted_memory.is_deleted

    @pytest.mark.asyncio
    async def test_memory_export_import(self, memory_pyramid, temp_dir):
        """Test memory export and import functionality"""
        # Store test memories
        test_memories = [
            ("Memory export test 1", MemoryType.USER_REQUEST, 0.8),
            ("Memory export test 2", MemoryType.IMPLEMENTATION, 0.7),
            ("Memory export test 3", MemoryType.DECISION, 0.9)
        ]

        original_ids = []
        for content, mem_type, importance in test_memories:
            memory_id = await memory_pyramid.store_memory(
                content=content,
                memory_type=mem_type,
                importance=importance
            )
            original_ids.append(memory_id)

        # Export memories
        export_file = temp_dir / "memory_export.json"
        await memory_pyramid.export_memories(str(export_file))

        assert export_file.exists()

        # Verify export content
        with open(export_file) as f:
            export_data = json.load(f)

        assert "memories" in export_data
        assert "metadata" in export_data
        assert len(export_data["memories"]) == 3

        # Clear current memories
        await memory_pyramid.cleanup()

        # Re-initialize and import
        await memory_pyramid.initialize()
        imported_count = await memory_pyramid.import_memories(str(export_file))

        assert imported_count == 3

        # Verify imported memories
        all_memories = await memory_pyramid.get_all_memories()
        assert len(all_memories) == 3

        imported_contents = [m.content for m in all_memories]
        original_contents = [content for content, _, _ in test_memories]

        for original_content in original_contents:
            assert original_content in imported_contents

    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, memory_pyramid):
        """Test thread-safe concurrent memory operations"""
        async def store_memories_batch(batch_id: int):
            """Store a batch of memories concurrently"""
            memory_ids = []
            for i in range(5):
                memory_id = await memory_pyramid.store_memory(
                    content=f"Batch {batch_id} - Memory {i}",
                    memory_type=MemoryType.IMPLEMENTATION,
                    importance=0.5
                )
                memory_ids.append(memory_id)
            return memory_ids

        # Run concurrent memory storage operations
        tasks = [store_memories_batch(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        # Verify all memories were stored correctly
        all_memory_ids = [memory_id for batch in results for memory_id in batch]
        assert len(all_memory_ids) == 15
        assert len(set(all_memory_ids)) == 15  # All IDs should be unique

        # Verify all memories can be retrieved
        for memory_id in all_memory_ids:
            memory = await memory_pyramid.retrieve_memory(memory_id)
            assert memory is not None

    @pytest.mark.asyncio
    async def test_memory_tier_promotion(self, memory_pyramid):
        """Test memory promotion between tiers based on access patterns"""
        # Store memory in short-term
        memory_id = await memory_pyramid.store_memory(
            content="Frequently accessed implementation detail",
            memory_type=MemoryType.IMPLEMENTATION,
            importance=0.6,
            tier=MemoryTier.SHORT_TERM
        )

        # Simulate frequent access
        for _ in range(10):
            await memory_pyramid.access_memory(memory_id)

        # Trigger tier evaluation
        await memory_pyramid.evaluate_memory_tiers()

        # Memory should be promoted to working tier due to frequent access
        memory = await memory_pyramid.retrieve_memory(memory_id)
        working_memories = await memory_pyramid.get_memories_by_tier(MemoryTier.WORKING)
        working_ids = [m.memory_id for m in working_memories]

        assert memory.access_count == 10
        assert memory_id in working_ids

    @pytest.mark.asyncio
    async def test_memory_pyramid_performance(self, memory_pyramid):
        """Test memory pyramid performance with large dataset"""
        import time

        # Store large number of memories
        start_time = time.time()
        memory_ids = []

        for i in range(100):
            memory_id = await memory_pyramid.store_memory(
                content=f"Performance test memory {i}",
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.5 + (i % 50) / 100  # Varying importance
            )
            memory_ids.append(memory_id)

        storage_time = time.time() - start_time

        # Test search performance
        start_time = time.time()
        search_results = await memory_pyramid.search_memories("test")
        search_time = time.time() - start_time

        # Test retrieval performance
        start_time = time.time()
        for memory_id in memory_ids[:10]:  # Test first 10
            await memory_pyramid.retrieve_memory(memory_id)
        retrieval_time = time.time() - start_time

        # Performance assertions (adjust thresholds based on requirements)
        assert storage_time < 30.0  # Should store 100 memories in under 30 seconds
        assert search_time < 1.0    # Search should be under 1 second
        assert retrieval_time < 1.0 # Retrieving 10 memories should be under 1 second

        # Verify all memories stored correctly
        assert len(memory_ids) == 100
        all_memories = await memory_pyramid.get_all_memories()
        assert len(all_memories) == 100


@pytest.mark.unit
@pytest.mark.gpt5
@pytest.mark.memory
class TestMemoryNode:
    """Test suite for MemoryNode class"""

    def test_memory_node_creation(self):
        """Test memory node creation and initialization"""
        node = MemoryNode(
            content="Test memory content",
            memory_type=MemoryType.IMPLEMENTATION,
            importance=0.8,
            tier=MemoryTier.WORKING
        )

        assert node.content == "Test memory content"
        assert node.memory_type == MemoryType.IMPLEMENTATION
        assert node.importance == 0.8
        assert node.tier == MemoryTier.WORKING
        assert node.access_count == 0
        assert not node.compressed
        assert not node.is_deleted
        assert node.created_at is not None

    def test_memory_node_serialization(self):
        """Test memory node to/from dictionary conversion"""
        node = MemoryNode(
            content="Serialization test",
            memory_type=MemoryType.DECISION,
            importance=0.7,
            tier=MemoryTier.SHORT_TERM
        )

        # Convert to dictionary
        node_dict = node.to_dict()

        assert isinstance(node_dict, dict)
        assert node_dict["content"] == "Serialization test"
        assert node_dict["memory_type"] == MemoryType.DECISION.value
        assert node_dict["importance"] == 0.7
        assert node_dict["tier"] == MemoryTier.SHORT_TERM.value

        # Convert back from dictionary
        restored_node = MemoryNode.from_dict(node_dict)

        assert restored_node.content == node.content
        assert restored_node.memory_type == node.memory_type
        assert restored_node.importance == node.importance
        assert restored_node.tier == node.tier

    def test_memory_node_update_access(self):
        """Test memory node access tracking"""
        node = MemoryNode(
            content="Access tracking test",
            memory_type=MemoryType.USER_REQUEST,
            importance=0.6
        )

        initial_access_count = node.access_count
        initial_last_accessed = node.last_accessed_at

        # Update access
        node.update_access()

        assert node.access_count == initial_access_count + 1
        assert node.last_accessed_at > initial_last_accessed

    def test_memory_node_importance_decay(self):
        """Test memory importance decay over time"""
        node = MemoryNode(
            content="Importance decay test",
            memory_type=MemoryType.OPTIMIZATION,
            importance=0.9
        )

        # Mock old creation time
        with patch.object(node, 'created_at', datetime.now() - timedelta(days=30)):
            decayed_importance = node.calculate_decayed_importance(decay_rate=0.01)

            assert decayed_importance < node.importance
            assert decayed_importance > 0


@pytest.mark.unit
@pytest.mark.gpt5
@pytest.mark.memory
class TestMemoryTierEnums:
    """Test memory tier and type enumerations"""

    def test_memory_tier_enum(self):
        """Test memory tier enumeration values"""
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.SHORT_TERM.value == "short_term"
        assert MemoryTier.LONG_TERM.value == "long_term"
        assert MemoryTier.ARCHIVE.value == "archive"

    def test_memory_type_enum(self):
        """Test memory type enumeration values"""
        expected_types = [
            "user_request",
            "implementation",
            "decision",
            "bug_fix",
            "optimization",
            "improvement",
            "error",
            "success"
        ]

        for memory_type in MemoryType:
            assert memory_type.value in expected_types