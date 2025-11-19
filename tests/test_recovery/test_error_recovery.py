"""
Tests for Error Recovery System

Validates that the system can:
- Detect errors and failures
- Apply learned solutions from error graph
- Recover gracefully without data loss
- Resume operations after recovery
"""

import pytest
import asyncio
from datetime import datetime
from core.memory.error_knowledge_graph import ErrorKnowledgeGraph


class TestErrorRecovery:
    """Test suite for error recovery workflows"""

    @pytest.fixture
    async def error_graph(self, tmp_path):
        """Create ErrorKnowledgeGraph for recovery testing"""
        graph = ErrorKnowledgeGraph(storage_path=str(tmp_path))
        await graph.initialize()
        return graph

    @pytest.fixture
    def sample_error(self):
        """Sample error for testing"""
        return {
            'type': 'ImportError',
            'message': 'No module named requests',
            'stack_trace': 'File test.py, line 1\nimport requests',
            'context': {'file': 'test.py', 'line': 1},
            'timestamp': datetime.now().isoformat()
        }

    @pytest.fixture
    def sample_solution(self):
        """Sample solution for error"""
        return {
            'description': 'Install requests package',
            'actions': ['pip install requests'],
            'success_rate': 0.95
        }

    @pytest.mark.asyncio
    async def test_record_error(self, error_graph, sample_error):
        """Test recording an error"""
        error_id = await error_graph.add_error(sample_error)

        assert error_id is not None

    @pytest.mark.asyncio
    async def test_find_solution(self, error_graph, sample_error, sample_solution):
        """Test finding solution for known error"""
        # Record error with solution
        error_id = await error_graph.add_error(sample_error)
        await error_graph.add_solution(error_id, sample_solution)

        # Find solution for similar error
        similar_error = {
            'type': 'ImportError',
            'message': 'No module named pandas'
        }

        solutions = await error_graph.find_solutions(similar_error)

        assert len(solutions) > 0
        assert 'pip install' in solutions[0]['actions'][0]

    @pytest.mark.asyncio
    async def test_automatic_recovery_attempt(self, error_graph, sample_error, sample_solution):
        """Test automatic recovery using learned solutions"""
        # Set up error with known solution
        error_id = await error_graph.add_error(sample_error)
        await error_graph.add_solution(error_id, sample_solution)

        # Simulate encountering similar error
        recovery_result = await error_graph.attempt_recovery(sample_error)

        assert recovery_result is not None
        assert recovery_result['attempted'] is True
        assert 'solution' in recovery_result

    @pytest.mark.asyncio
    async def test_recovery_failure_handling(self, error_graph, sample_error):
        """Test handling when recovery fails"""
        # Add error without solution
        error_id = await error_graph.add_error(sample_error)

        # Attempt recovery
        result = await error_graph.attempt_recovery(sample_error)

        # Should handle gracefully
        assert result['attempted'] is False or result['success'] is False
        assert 'fallback_action' in result

    @pytest.mark.asyncio
    async def test_error_pattern_learning(self, error_graph):
        """Test system learns patterns from repeated errors"""
        # Record same type of error multiple times
        for i in range(5):
            error = {
                'type': 'ValueError',
                'message': f'Invalid value: {i}',
                'context': {}
            }
            await error_graph.add_error(error)

        # Check if pattern detected
        patterns = await error_graph.get_patterns()

        assert len(patterns) > 0
        assert any(p['error_type'] == 'ValueError' for p in patterns)

    @pytest.mark.asyncio
    async def test_solution_effectiveness_tracking(self, error_graph, sample_error, sample_solution):
        """Test tracking solution effectiveness over time"""
        error_id = await error_graph.add_error(sample_error)
        solution_id = await error_graph.add_solution(error_id, sample_solution)

        # Record successes and failures
        await error_graph.record_solution_result(solution_id, success=True)
        await error_graph.record_solution_result(solution_id, success=True)
        await error_graph.record_solution_result(solution_id, success=False)

        # Check effectiveness
        solution = await error_graph.get_solution(solution_id)

        assert solution['total_attempts'] == 3
        assert solution['success_count'] == 2
        assert solution['success_rate'] == pytest.approx(0.67, rel=0.1)

    @pytest.mark.asyncio
    async def test_cascading_error_recovery(self, error_graph):
        """Test recovering from cascading errors"""
        # Simulate chain of related errors
        errors = [
            {'type': 'ConfigError', 'message': 'Missing config file'},
            {'type': 'DatabaseError', 'message': 'Connection failed'},
            {'type': 'RuntimeError', 'message': 'Service unavailable'}
        ]

        error_ids = []
        for error in errors:
            error_id = await error_graph.add_error(error)
            error_ids.append(error_id)

        # Link errors as cascade
        await error_graph.link_errors(error_ids, relationship='cascade')

        # Resolve root cause
        await error_graph.resolve_error(error_ids[0])

        # Should mark related errors as potentially resolved
        status = await error_graph.get_cascade_status(error_ids)

        assert status['root_resolved'] is True

    @pytest.mark.asyncio
    async def test_recovery_state_persistence(self, error_graph, sample_error, tmp_path):
        """Test recovery state persists across restarts"""
        # Record error and attempt recovery
        error_id = await error_graph.add_error(sample_error)
        await error_graph.mark_recovery_in_progress(error_id)

        # Restart system
        await error_graph.shutdown()

        new_graph = ErrorKnowledgeGraph(storage_path=str(tmp_path))
        await new_graph.initialize()

        # Check recovery state persisted
        error = await new_graph.get_error(error_id)

        assert error['recovery_status'] == 'in_progress'

    @pytest.mark.asyncio
    async def test_partial_recovery(self, error_graph):
        """Test handling partial recoveries"""
        error = {
            'type': 'MultipleErrors',
            'sub_errors': [
                {'type': 'Error1', 'message': 'First error'},
                {'type': 'Error2', 'message': 'Second error'}
            ]
        }

        error_id = await error_graph.add_error(error)

        # Resolve only first sub-error
        await error_graph.resolve_sub_error(error_id, sub_index=0)

        # Check partial resolution
        status = await error_graph.get_resolution_status(error_id)

        assert status['total_errors'] == 2
        assert status['resolved_errors'] == 1
        assert status['is_fully_resolved'] is False

    @pytest.mark.asyncio
    async def test_recovery_timeout(self, error_graph, sample_error):
        """Test recovery attempts time out appropriately"""
        error_id = await error_graph.add_error(sample_error)

        # Attempt recovery with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                error_graph.attempt_complex_recovery(error_id),
                timeout=1.0  # 1 second timeout
            )

        # Check error marked as timeout
        error = await error_graph.get_error(error_id)

        assert error['recovery_status'] in ['timeout', 'failed']

    @pytest.mark.asyncio
    async def test_recovery_rollback(self, error_graph, sample_error):
        """Test rolling back failed recovery attempts"""
        error_id = await error_graph.add_error(sample_error)

        # Attempt recovery that fails
        recovery_result = await error_graph.attempt_recovery_with_rollback(
            error_id,
            checkpoint_id='pre_recovery_checkpoint'
        )

        if not recovery_result['success']:
            # Should have rolled back
            assert recovery_result['rolled_back'] is True
            assert recovery_result['checkpoint_restored'] == 'pre_recovery_checkpoint'
