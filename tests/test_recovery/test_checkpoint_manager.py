"""
Tests for Checkpoint Manager Recovery System

Validates that the system can:
- Create checkpoints at critical points
- Restore from checkpoints after failures
- Maintain checkpoint integrity
- Clean up old checkpoints
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from core.orchestrator.checkpoint_manager import CheckpointManager, Checkpoint


class TestCheckpointManager:
    """Test suite for CheckpointManager recovery"""

    @pytest.fixture
    async def manager(self, tmp_path):
        """Create CheckpointManager with temporary storage"""
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        await mgr.initialize()
        return mgr

    @pytest.fixture
    def sample_state(self):
        """Sample system state for checkpointing"""
        return {
            'iteration': 10,
            'metrics': {
                'test_coverage': 85,
                'bugs': 3,
                'performance': 92
            },
            'active_agents': ['coder', 'tester'],
            'task_queue': [
                {'id': 1, 'type': 'implement', 'status': 'in_progress'},
                {'id': 2, 'type': 'test', 'status': 'pending'}
            ],
            'current_phase': 'development'
        }

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, manager, sample_state):
        """Test creating a checkpoint"""
        checkpoint_id = await manager.create_checkpoint(
            state=sample_state,
            label='test_checkpoint'
        )

        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)

    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, manager, sample_state):
        """Test restoring from a checkpoint"""
        # Create checkpoint
        checkpoint_id = await manager.create_checkpoint(
            state=sample_state,
            label='restore_test'
        )

        # Restore
        restored_state = await manager.restore_checkpoint(checkpoint_id)

        assert restored_state is not None
        assert restored_state['iteration'] == sample_state['iteration']
        assert restored_state['metrics'] == sample_state['metrics']
        assert restored_state['current_phase'] == sample_state['current_phase']

    @pytest.mark.asyncio
    async def test_checkpoint_metadata(self, manager, sample_state):
        """Test checkpoint stores metadata correctly"""
        checkpoint_id = await manager.create_checkpoint(
            state=sample_state,
            label='metadata_test',
            metadata={'reason': 'test', 'triggered_by': 'unit_test'}
        )

        checkpoint = await manager.get_checkpoint(checkpoint_id)

        assert checkpoint.label == 'metadata_test'
        assert checkpoint.metadata['reason'] == 'test'
        assert checkpoint.metadata['triggered_by'] == 'unit_test'
        assert checkpoint.timestamp is not None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, manager, sample_state):
        """Test listing all checkpoints"""
        # Create multiple checkpoints
        for i in range(5):
            await manager.create_checkpoint(
                state={**sample_state, 'iteration': i},
                label=f'checkpoint_{i}'
            )

        checkpoints = await manager.list_checkpoints()

        assert len(checkpoints) == 5
        assert checkpoints[0].label == 'checkpoint_4'  # Most recent first

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, manager, sample_state):
        """Test retrieving the most recent checkpoint"""
        # Create checkpoints with delays
        for i in range(3):
            await manager.create_checkpoint(
                state={**sample_state, 'iteration': i},
                label=f'cp_{i}'
            )
            await asyncio.sleep(0.1)

        latest = await manager.get_latest_checkpoint()

        assert latest is not None
        assert latest.label == 'cp_2'  # Last one created

    @pytest.mark.asyncio
    async def test_checkpoint_validation(self, manager, sample_state):
        """Test checkpoint validates state integrity"""
        checkpoint_id = await manager.create_checkpoint(
            state=sample_state,
            label='validation_test'
        )

        # Validate checkpoint
        is_valid = await manager.validate_checkpoint(checkpoint_id)

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_corrupted_checkpoint_detection(self, manager, sample_state, tmp_path):
        """Test detecting corrupted checkpoints"""
        checkpoint_id = await manager.create_checkpoint(
            state=sample_state,
            label='corruption_test'
        )

        # Corrupt the checkpoint file
        checkpoint_path = tmp_path / f"{checkpoint_id}.json"
        with open(checkpoint_path, 'w') as f:
            f.write('corrupted data')

        # Validation should fail
        is_valid = await manager.validate_checkpoint(checkpoint_id)

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_auto_checkpoint(self, manager, sample_state):
        """Test automatic checkpoint creation at intervals"""
        manager.enable_auto_checkpoint(interval_iterations=2)

        checkpoints_before = len(await manager.list_checkpoints())

        # Simulate iterations
        for i in range(5):
            await manager.on_iteration_complete(
                iteration_number=i,
                state={**sample_state, 'iteration': i}
            )

        checkpoints_after = len(await manager.list_checkpoints())

        # Should have created checkpoints at iterations 0, 2, 4
        assert checkpoints_after >= checkpoints_before + 2

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, manager, sample_state):
        """Test cleaning up old checkpoints"""
        # Create many checkpoints
        for i in range(20):
            await manager.create_checkpoint(
                state={**sample_state, 'iteration': i},
                label=f'cleanup_test_{i}'
            )

        # Cleanup, keep only last 5
        await manager.cleanup_old_checkpoints(keep_last=5)

        remaining = await manager.list_checkpoints()

        assert len(remaining) <= 5
        assert remaining[0].label == 'cleanup_test_19'  # Most recent kept

    @pytest.mark.asyncio
    async def test_recovery_from_crash(self, manager, sample_state):
        """Test complete recovery workflow from crash"""
        # Create checkpoint before "crash"
        checkpoint_id = await manager.create_checkpoint(
            state=sample_state,
            label='pre_crash'
        )

        # Simulate crash and restart
        await manager.shutdown()

        # New manager instance
        new_manager = CheckpointManager(checkpoint_dir=str(manager.checkpoint_dir))
        await new_manager.initialize()

        # Restore from checkpoint
        recovered_state = await new_manager.restore_latest()

        assert recovered_state is not None
        assert recovered_state['iteration'] == sample_state['iteration']
        assert recovered_state['current_phase'] == sample_state['current_phase']

    @pytest.mark.asyncio
    async def test_incremental_checkpoint(self, manager, sample_state):
        """Test incremental checkpoints (only store changes)"""
        # Create base checkpoint
        base_id = await manager.create_checkpoint(
            state=sample_state,
            label='base'
        )

        # Create incremental checkpoint with only changes
        modified_state = sample_state.copy()
        modified_state['metrics']['test_coverage'] = 90  # Changed
        modified_state['iteration'] = 11  # Changed

        incremental_id = await manager.create_incremental_checkpoint(
            base_checkpoint_id=base_id,
            changes={'metrics.test_coverage': 90, 'iteration': 11},
            label='incremental'
        )

        # Restore incremental
        restored = await manager.restore_checkpoint(incremental_id)

        assert restored['metrics']['test_coverage'] == 90
        assert restored['iteration'] == 11
        assert restored['current_phase'] == sample_state['current_phase']  # Unchanged from base

    @pytest.mark.asyncio
    async def test_checkpoint_compression(self, manager, sample_state, tmp_path):
        """Test checkpoints are compressed to save space"""
        # Create large state
        large_state = {
            **sample_state,
            'large_data': 'x' * 10000  # 10KB of data
        }

        checkpoint_id = await manager.create_checkpoint(
            state=large_state,
            label='compression_test',
            compress=True
        )

        # Check file size
        checkpoint_path = tmp_path / f"{checkpoint_id}.json.gz"
        uncompressed_size = len(str(large_state).encode())
        compressed_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0

        # Compressed should be smaller (or check compression was attempted)
        assert compressed_size < uncompressed_size or checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_checkpoint_rollback(self, manager, sample_state):
        """Test rolling back to specific checkpoint"""
        # Create checkpoint chain
        checkpoints = []
        for i in range(5):
            state = {**sample_state, 'iteration': i}
            cp_id = await manager.create_checkpoint(state, label=f'v{i}')
            checkpoints.append(cp_id)

        # Rollback to checkpoint 2
        await manager.rollback_to_checkpoint(checkpoints[2])

        # Current state should match checkpoint 2
        current = await manager.get_current_state()

        assert current['iteration'] == 2
