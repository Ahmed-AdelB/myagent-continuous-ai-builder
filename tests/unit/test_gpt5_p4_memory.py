"""
Unit tests for GPT-5 Priority 4: Hierarchical Memory Pyramid
Tests the 4-tier memory architecture with automatic consolidation
"""

import pytest
import pytest
import pytest_asyncio
import asyncio
import tempfile
import json
from datetime import datetime, timedelta, timezone
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

    @pytest_asyncio.fixture
    async def memory_pyramid_instance(self, temp_database):
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
    async def test_store_memory_working_tier(self, memory_pyramid_instance):
        """Test storing memory in working tier"""
        content = "User requested fibonacci function implementation"
        memory_type = MemoryType.USER_REQUEST

        memory_id = await memory_pyramid_instance.store_memory(
            content=content,
            memory_type=memory_type,
            importance=0.8
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)

        # Verify memory is in working tier
        # Note: In a real implementation we would query the database or check internal state
        # For now we assume store_memory works if it returns an ID
        working_memories = await memory_pyramid_instance.get_memories_by_tier(MemoryTier.WORKING)
        assert len(working_memories) == 1
        assert working_memories[0].content == content
        assert working_memories[0].memory_type == memory_type
        assert working_memories[0].importance == 0.8

    @pytest.mark.asyncio
    async def test_store_multiple_memories(self, memory_pyramid_instance):
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
            memory_id = await memory_pyramid_instance.store_memory(
                content=memory_data["content"],
                memory_type=memory_data["memory_type"],
                importance=memory_data["importance"],
                tier=memory_data["tier"]
            )
            memory_ids.append(memory_id)

        # Verify all memories stored
        assert len(memory_ids) == 3
        assert all(mid is not None for mid in memory_ids)
        assert all(isinstance(mid, str) for mid in memory_ids)

        # Verify memories in correct tiers
        working_memories = await memory_pyramid_instance.get_memories_by_tier(MemoryTier.WORKING)
        short_term_memories = await memory_pyramid_instance.get_memories_by_tier(MemoryTier.SHORT_TERM)
        long_term_memories = await memory_pyramid_instance.get_memories_by_tier(MemoryTier.LONG_TERM)

        assert len(working_memories) == 1
        assert len(short_term_memories) == 1
        assert len(long_term_memories) == 1

    @pytest.mark.asyncio
    async def test_retrieve_memory_by_id(self, memory_pyramid_instance):
        """Test retrieving specific memory by ID"""
        content = "Test memory for retrieval"
        memory_type = MemoryType.DECISION

        memory_id = await memory_pyramid_instance.store_memory(
            content=content,
            memory_type=memory_type,
            importance=0.6
        )

        # Retrieve memory
        retrieved_memory = await memory_pyramid_instance.retrieve_memory(memory_id)

        assert retrieved_memory is not None
        assert retrieved_memory.node_id == memory_id
        assert retrieved_memory.content == content
        assert retrieved_memory.memory_type == memory_type
        assert retrieved_memory.importance == 0.6

    @pytest.mark.asyncio
    async def test_search_memories_by_content(self, memory_pyramid_instance):
        """Test searching memories by content"""
        # Store test memories
        test_memories = [
            "Implement fibonacci function using recursion",
            "Optimize fibonacci with memoization",
            "Create unit tests for calculator module",
            "Debug performance issues in sorting algorithm"
        ]

        for content in test_memories:
            await memory_pyramid_instance.store_memory(
                content=content,
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.7
            )

        # Search for fibonacci-related memories
        fibonacci_memories = await memory_pyramid_instance.search_memories("fibonacci")
        assert len(fibonacci_memories) == 2

        # Search for testing-related memories
        test_related = await memory_pyramid_instance.search_memories("test")
        assert len(test_related) == 1

        # Search for non-existent content
        no_results = await memory_pyramid_instance.search_memories("nonexistent")
        assert len(no_results) == 0

    @pytest.mark.asyncio
    async def test_search_memories_by_type(self, memory_pyramid_instance):
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
            await memory_pyramid_instance.store_memory(
                content=content,
                memory_type=mem_type,
                importance=0.7
            )

        # Search by specific types
        implementations = await memory_pyramid_instance.search_memories_by_type(MemoryType.IMPLEMENTATION)
        decisions = await memory_pyramid_instance.search_memories_by_type(MemoryType.DECISION)
        bug_fixes = await memory_pyramid_instance.search_memories_by_type(MemoryType.BUG_FIX)

        assert len(implementations) == 1
        assert len(decisions) == 1
        assert len(bug_fixes) == 1

    @pytest.mark.asyncio
    async def test_memory_consolidation_by_count(self, memory_pyramid_instance):
        """Test automatic memory consolidation when working memory is full"""
        # Set low threshold for testing
        pyramid = memory_pyramid_instance
        pyramid.working_memory_threshold = 3

        # Fill working memory beyond threshold
        for i in range(5):
            memory_id = await pyramid.store_memory(
                content=f"Working memory item {i}",
                memory_type=MemoryType.USER_REQUEST,
                importance=0.5 + (i * 0.1)  # Varying importance
            )
            # Make memory old enough to be consolidated (older than 1 hour)
            memory = await pyramid.retrieve_memory(memory_id)
            memory.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
            # Also increase access count to meet threshold (2)
            memory.access_count = 3
            await pyramid.update_memory(memory)

        # Trigger consolidation
        await pyramid.consolidate_memories(MemoryTier.WORKING)

        # Verify consolidation occurred
        working_memories = await pyramid.get_memories_by_tier(MemoryTier.WORKING)
        short_term_memories = await pyramid.get_memories_by_tier(MemoryTier.SHORT_TERM)

        # High importance memories should remain in working
        # Lower importance should move to short-term
        assert len(working_memories) <= pyramid.working_memory_threshold
        assert len(short_term_memories) > 0

    @pytest.mark.asyncio
    async def test_memory_consolidation_by_time(self, memory_pyramid_instance):
        """Test time-based memory consolidation"""
        pyramid = memory_pyramid_instance

        # Create old memory (simulate)
        old_content = "Old memory for consolidation test"
        memory_id = await pyramid.store_memory(
            content=old_content,
            memory_type=MemoryType.IMPLEMENTATION,
            importance=0.6
        )

        # Access memory to meet threshold (needs 2 accesses)
        await pyramid.retrieve_memory(memory_id)
        await pyramid.retrieve_memory(memory_id)

        # Manually update timestamp to simulate age
        with patch('core.memory_pyramid.hierarchical_memory_pyramid.datetime') as mock_datetime:
            # Mock future timestamp to simulate age (25 hours later)
            future_time = datetime.now(timezone.utc) + timedelta(hours=25)
            mock_datetime.now.return_value = future_time
            # Also mock utcnow if used
            mock_datetime.utcnow.return_value = future_time.replace(tzinfo=None)

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
    async def test_memory_compression(self, memory_pyramid_instance):
        """Test memory compression for archive tier"""
        pyramid = memory_pyramid_instance

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
        assert "fibonacci" in compressed_memory.content["summary"]
        assert compressed_memory.tier == MemoryTier.LONG_TERM

        # Original memories should be marked as compressed
        for memory_id in memory_ids:
            memory = await pyramid.retrieve_memory(memory_id)
            assert memory.metadata.get('is_compressed') is True

    @pytest.mark.asyncio
    async def test_memory_importance_scoring(self, memory_pyramid_instance):
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
            memory_id = await memory_pyramid_instance.store_memory(
                content=memory_data["content"],
                memory_type=memory_data["memory_type"],
                importance=memory_data["importance"]
            )
            memory_ids.append(memory_id)

            # Simulate access count
            memory = await memory_pyramid_instance.retrieve_memory(memory_id)
            memory.access_count = memory_data["access_count"]
            await memory_pyramid_instance.update_memory(memory)

        # Get memories by importance
        important_memories = await memory_pyramid_instance.get_memories_by_importance(min_importance=0.7)
        assert len(important_memories) == 2  # Security bug and core algorithm

        # Get most accessed memories
        accessed_memories = await memory_pyramid_instance.get_most_accessed_memories(limit=2)
        assert len(accessed_memories) == 2
        assert accessed_memories[0].access_count >= accessed_memories[1].access_count

    @pytest.mark.asyncio
    async def test_memory_relationships(self, memory_pyramid_instance):
        """Test memory relationship tracking"""
        # Create parent memory
        parent_id = await memory_pyramid_instance.store_memory(
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
            child_id = await memory_pyramid_instance.store_memory(
                content=content,
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.7,
                parent_memory_id=parent_id
            )
            child_ids.append(child_id)

        # Verify relationships
        children = await memory_pyramid_instance.get_child_memories(parent_id)
        assert len(children) == 3

        # Verify parent relationship
        for child_id in child_ids:
            child_memory = await memory_pyramid_instance.retrieve_memory(child_id)
            assert parent_id in child_memory.parent_nodes

    @pytest.mark.asyncio
    async def test_memory_statistics(self, memory_pyramid_instance):
        """Test memory usage statistics"""
        # Add various memories
        for i in range(10):
            await memory_pyramid_instance.store_memory(
                content=f"Test memory {i}",
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.5 + (i * 0.05),
                tier=MemoryTier.WORKING if i < 3 else MemoryTier.SHORT_TERM
            )

        # Get statistics
        stats = await memory_pyramid_instance.get_memory_statistics()
    
        assert stats["total_memories"] == 10
        assert stats["tier_distributions"][MemoryTier.WORKING.value] == 3
        assert stats["tier_distributions"][MemoryTier.SHORT_TERM.value] == 7
        assert "tier_distributions" in stats
        assert "memory_type_distribution" in stats

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_pyramid_instance):
        """Test memory cleanup and garbage collection"""
        # Create low-importance memories
        low_importance_ids = []
        for i in range(5):
            memory_id = await memory_pyramid_instance.store_memory(
                content=f"Low importance memory {i}",
                memory_type=MemoryType.OPTIMIZATION,
                importance=0.1,
                tier=MemoryTier.ARCHIVE
            )
            low_importance_ids.append(memory_id)

        # Create high-importance memory
        high_importance_id = await memory_pyramid_instance.store_memory(
            content="Critical system information",
            memory_type=MemoryType.DECISION,
            importance=0.95,
            tier=MemoryTier.LONG_TERM
        )

        # Perform cleanup (remove memories below threshold)
        await memory_pyramid_instance.cleanup_low_importance()
        
        # Verify cleanup
        # Note: cleanup_low_importance doesn't return count, so we check existence
        remaining_memories = await memory_pyramid_instance.get_memories_by_tier(MemoryTier.ARCHIVE)
        # Should only have high importance ones or empty if all were low
        # In this test, we put low importance in ARCHIVE and high in LONG_TERM
        # So ARCHIVE should be empty
        assert len(remaining_memories) == 0

        # Verify high-importance memory remains
        high_importance_memory = await memory_pyramid_instance.retrieve_memory(high_importance_id)
        assert high_importance_memory is not None

        # Verify low-importance memories removed
        for memory_id in low_importance_ids:
            deleted_memory = await memory_pyramid_instance.retrieve_memory(memory_id)
            assert deleted_memory is None or deleted_memory.is_deleted

    @pytest.mark.asyncio
    async def test_memory_export_import(self, memory_pyramid_instance, temp_dir):
        """Test memory export and import functionality"""
        # Store test memories
        test_memories = [
            ("Memory export test 1", MemoryType.USER_REQUEST, 0.8),
            ("Memory export test 2", MemoryType.IMPLEMENTATION, 0.7),
            ("Memory export test 3", MemoryType.DECISION, 0.9)
        ]

        original_ids = []
        for content, mem_type, importance in test_memories:
            memory_id = await memory_pyramid_instance.store_memory(
                content=content,
                memory_type=mem_type,
                importance=importance
            )
            original_ids.append(memory_id)

        # Export memories
        export_path = temp_dir / "memory_export.json"
        await memory_pyramid_instance.export_memories(str(export_path))

        assert export_path.exists()

        # Verify export content
        with open(export_path) as f:
            export_data = json.load(f)
    
        assert isinstance(export_data, list)
        assert len(export_data) == 3
        # Check content of first item
        assert "content" in export_data[0]
        assert "metadata" in export_data[0]

        # Clear current memories
        await memory_pyramid_instance.cleanup()

        # Re-initialize and import
        await memory_pyramid_instance.initialize()
        imported_count = await memory_pyramid_instance.import_memories(export_path)

        assert imported_count == 3

        # Verify imported memories
        stats = await memory_pyramid_instance.get_memory_statistics()
        assert stats['total_memories'] == 3
        
        all_memories = []
        for tier in MemoryTier:
            tier_memories = await memory_pyramid_instance.get_memories_by_tier(tier)
            all_memories.extend(tier_memories)
            
        assert len(all_memories) == 3

        imported_contents = [m.content for m in all_memories]
        original_contents = [content for content, _, _ in test_memories]

        for original_content in original_contents:
            assert original_content in imported_contents

        # Create new pyramid and import
        new_pyramid = HierarchicalMemoryPyramid(database_path=temp_dir / "new_db.db")
        await new_pyramid.initialize()
        imported_count_new = await new_pyramid.import_memories(export_path)
        assert imported_count_new == 3
        await new_pyramid.cleanup()


    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, memory_pyramid_instance):
        """Test thread-safe concurrent memory operations"""
        async def store_memories_batch(batch_id: int):
            """Store a batch of memories concurrently"""
            memory_ids = []
            for i in range(5):
                memory_id = await memory_pyramid_instance.store_memory(
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
            memory = await memory_pyramid_instance.retrieve_memory(memory_id)
            assert memory is not None

    @pytest.mark.asyncio
    async def test_memory_tier_promotion(self, memory_pyramid_instance):
        """Test memory promotion between tiers based on access patterns"""
        # Store memory in short-term
        memory_id = await memory_pyramid_instance.store_memory(
            content="Frequently accessed implementation detail",
            memory_type=MemoryType.IMPLEMENTATION,
            importance=0.6,
            tier=MemoryTier.SHORT_TERM
        )

        # Access memory multiple times to trigger promotion
        for _ in range(10):
            await memory_pyramid_instance.retrieve_memory(memory_id)

        # Trigger tier evaluation (happens automatically on retrieve, but we can verify)
        # await memory_pyramid_instance.evaluate_memory_tiers() # Removed as it's automatic

        # Memory should be promoted to working tier due to frequent access
        memory = await memory_pyramid_instance.retrieve_memory(memory_id)
        working_memories = await memory_pyramid_instance.get_memories_by_tier(MemoryTier.WORKING)
        working_ids = [m.node_id for m in working_memories]

        assert memory.access_count == 12
        assert memory_id in working_ids

    @pytest.mark.asyncio
    async def test_memory_pyramid_performance(self, memory_pyramid_instance):
        """Test memory pyramid performance with large dataset"""
        import time

        # Store large number of memories
        start_time = time.time()
        memory_ids = []

        for i in range(100):
            memory_id = await memory_pyramid_instance.store_memory(
                content=f"Performance test memory {i}",
                memory_type=MemoryType.IMPLEMENTATION,
                importance=0.5 + (i % 50) / 100  # Varying importance
            )
            memory_ids.append(memory_id)

        storage_time = time.time() - start_time

        # Test search performance
        start_time = time.time()
        search_results = await memory_pyramid_instance.search_memories("test")
        search_time = time.time() - start_time

        # Test retrieval performance
        start_time = time.time()
        for memory_id in memory_ids[:10]:  # Test first 10
            await memory_pyramid_instance.retrieve_memory(memory_id)
        retrieval_time = time.time() - start_time

        # Performance assertions (adjust thresholds based on requirements)
        assert storage_time < 30.0  # Should store 100 memories in under 30 seconds
        assert search_time < 1.0    # Search should be under 1 second
        assert retrieval_time < 1.0 # Retrieving 10 memories should be under 1 second

        # Verify all memories stored correctly
        assert len(memory_ids) == 100
        # all_memories = await memory_pyramid_instance.get_all_memories() # Method doesn't exist, checking count via stats
        stats = await memory_pyramid_instance.get_memory_statistics()
        assert stats['total_memories'] == 100


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

        # Manually set old creation time using the setter
        node.created_at = datetime.now(timezone.utc) - timedelta(days=30)
        
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
            "success",
            "episodic",
            "semantic",
            "procedural",
            "contextual"
        ]

        for memory_type in MemoryType:
            assert memory_type.value in expected_types