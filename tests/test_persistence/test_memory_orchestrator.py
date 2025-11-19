"""
Tests for Memory Orchestrator Persistence

Validates that memory orchestration correctly:
- Persists across sessions
- Maintains consistency between memory systems
- Handles concurrent access
- Recovers from failures
"""

import pytest
import asyncio
from datetime import datetime
from core.memory.memory_orchestrator import MemoryOrchestrator


class TestMemoryOrchestrator:
    """Test suite for MemoryOrchestrator persistence"""

    @pytest.fixture
    async def orchestrator(self, tmp_path):
        """Create MemoryOrchestrator with temporary storage"""
        orch = MemoryOrchestrator(storage_path=str(tmp_path))
        await orch.initialize()
        return orch

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initializes all memory systems"""
        assert orchestrator.is_initialized
        assert orchestrator.project_ledger is not None
        assert orchestrator.error_graph is not None
        assert orchestrator.vector_memory is not None

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, orchestrator):
        """Test storing and retrieving data"""
        test_data = {
            'key': 'test_key',
            'value': 'test_value',
            'metadata': {'type': 'test'}
        }

        # Store
        await orchestrator.store(
            key=test_data['key'],
            value=test_data['value'],
            metadata=test_data['metadata']
        )

        # Retrieve
        retrieved = await orchestrator.retrieve(test_data['key'])

        assert retrieved is not None
        assert retrieved['value'] == test_data['value']
        assert retrieved['metadata']['type'] == 'test'

    @pytest.mark.asyncio
    async def test_memory_synchronization(self, orchestrator):
        """Test synchronization between memory systems"""
        # Store in project ledger
        ledger_data = {
            'version': 1,
            'changes': 'Test change'
        }
        await orchestrator.project_ledger.record(ledger_data)

        # Should be reflected in orchestrator state
        state = await orchestrator.get_state()

        assert state['ledger_version'] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_access(self, orchestrator):
        """Test handling concurrent memory access"""
        async def store_data(i):
            return await orchestrator.store(
                key=f'key_{i}',
                value=f'value_{i}',
                metadata={'index': i}
            )

        # Store 20 items concurrently
        tasks = [store_data(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 20

        # All should be retrievable
        for i in range(20):
            retrieved = await orchestrator.retrieve(f'key_{i}')
            assert retrieved['value'] == f'value_{i}'

    @pytest.mark.asyncio
    async def test_persistence_across_sessions(self, orchestrator, tmp_path):
        """Test data persists across orchestrator restarts"""
        # Store data
        await orchestrator.store(
            key='persistent_key',
            value='persistent_value',
            metadata={}
        )

        # Close orchestrator
        await orchestrator.shutdown()

        # Create new orchestrator with same storage
        new_orchestrator = MemoryOrchestrator(storage_path=str(tmp_path))
        await new_orchestrator.initialize()

        # Data should still exist
        retrieved = await new_orchestrator.retrieve('persistent_key')
        assert retrieved is not None
        assert retrieved['value'] == 'persistent_value'

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, orchestrator):
        """Test memory cleanup removes old entries"""
        # Store many old entries
        for i in range(100):
            await orchestrator.store(
                key=f'old_key_{i}',
                value=f'old_value_{i}',
                metadata={'timestamp': datetime(2020, 1, 1).isoformat()}
            )

        # Run cleanup (keep recent only)
        await orchestrator.cleanup(keep_days=30)

        # Old entries should be removed
        result = await orchestrator.retrieve('old_key_0')
        assert result is None or result['value'] != 'old_value_0'

    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test orchestrator handles storage errors gracefully"""
        # Try to retrieve non-existent key
        result = await orchestrator.retrieve('nonexistent_key')
        assert result is None

        # Try to store with invalid data
        with pytest.raises(ValueError):
            await orchestrator.store(key='', value=None, metadata={})

    @pytest.mark.asyncio
    async def test_memory_consistency(self, orchestrator):
        """Test consistency between different memory systems"""
        # Store error in error graph
        error_data = {
            'type': 'TypeError',
            'message': 'Test error',
            'solution': 'Test solution'
        }
        await orchestrator.error_graph.add_error(error_data)

        # Store related version in ledger
        from core.memory.project_ledger import ProjectVersion
        version = ProjectVersion(
            version_number=1,
            timestamp=datetime.now(),
            files_changed=['test.py'],
            changes_summary='Fixed TypeError',
            metrics={},
            agent='debugger'
        )
        await orchestrator.project_ledger.record_version(version)

        # Query should show relationship
        related = await orchestrator.get_related_memories(
            error_type='TypeError'
        )

        assert len(related) > 0
        assert any('Fixed TypeError' in str(r) for r in related)
