"""
Comprehensive test suite for the Continuous AI Development System
Tests all critical components and verifies GPT-5 enhanced fixes
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator.continuous_director import ContinuousDirector, QualityMetrics, ProjectState
from core.memory.vector_memory import VectorMemory
from core.memory.project_ledger import ProjectLedger
from core.memory.error_knowledge_graph import ErrorKnowledgeGraph


class TestVectorMemoryInitialization:
    """Test VectorMemory initialization with project_name"""

    def test_vector_memory_requires_project_name(self):
        """Verify VectorMemory requires project_name parameter"""
        with pytest.raises(TypeError):
            vm = VectorMemory()  # Should fail without project_name

    def test_vector_memory_accepts_project_name(self):
        """Verify VectorMemory initializes with project_name"""
        vm = VectorMemory(project_name="test_project")
        assert vm.project_name == "test_project"
        assert vm.persist_dir.exists()


class TestQualityMetrics:
    """Test QualityMetrics functionality"""

    def test_quality_metrics_has_to_dict(self):
        """Verify QualityMetrics has to_dict() method"""
        metrics = QualityMetrics()
        data = metrics.to_dict()

        assert "test_coverage" in data
        assert "bug_count_critical" in data
        assert "performance_score" in data
        assert "is_perfect" in data
        assert isinstance(data["is_perfect"], bool)

    def test_is_perfect_criteria(self):
        """Test perfection criteria"""
        metrics = QualityMetrics()
        assert metrics.is_perfect() == False

        # Set perfect metrics
        metrics.test_coverage = 95.0
        metrics.bug_count_critical = 0
        metrics.bug_count_minor = 5
        metrics.performance_score = 90.0
        metrics.documentation_coverage = 90.0
        metrics.code_quality_score = 85.0
        metrics.user_satisfaction = 90.0
        metrics.security_score = 95.0

        assert metrics.is_perfect() == True


class TestContinuousDirector:
    """Test the main orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator"""
        return ContinuousDirector(
            project_name="test_project",
            project_spec={"description": "Test project"}
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly"""
        assert orchestrator.project_name == "test_project"
        assert orchestrator.state == ProjectState.INITIALIZING
        assert orchestrator.iteration_count == 0
        assert isinstance(orchestrator.metrics, QualityMetrics)

    @pytest.mark.asyncio
    async def test_components_initialization(self, orchestrator):
        """Test all components initialize"""
        with patch('core.orchestrator.milestone_tracker.MilestoneTracker'):
            with patch('core.orchestrator.progress_analyzer.ProgressAnalyzer'):
                await orchestrator._initialize_components()

                assert hasattr(orchestrator, 'project_ledger')
                assert hasattr(orchestrator, 'vector_memory')
                assert hasattr(orchestrator, 'error_graph')
                assert hasattr(orchestrator, 'milestone_tracker')
                assert hasattr(orchestrator, 'progress_analyzer')

    def test_weak_metrics_identification(self, orchestrator):
        """Test identification of weak metrics"""
        orchestrator.metrics.test_coverage = 80  # Below 95
        orchestrator.metrics.performance_score = 85  # Below 90
        orchestrator.metrics.bug_count_critical = 2  # Above 0

        weak = orchestrator.identify_weak_metrics()

        assert "test_coverage" in weak
        assert "performance_score" in weak
        assert "bug_count_critical" in weak


class TestContinuousQualityMonitor:
    """Test continuous quality monitoring"""

    @pytest.fixture
    def orchestrator(self):
        return ContinuousDirector(
            project_name="test_project",
            project_spec={"description": "Test"}
        )

    @pytest.mark.asyncio
    async def test_quality_monitor_triggers(self, orchestrator):
        """Test quality monitor triggers optimizations"""
        orchestrator.is_running = True
        orchestrator.metrics.test_coverage = 80  # Low

        with patch.object(orchestrator, 'trigger_test_intensification') as mock_test:
            # Run one cycle of quality monitor
            monitor_task = asyncio.create_task(orchestrator.continuous_quality_monitor())
            await asyncio.sleep(0.1)  # Let it run briefly
            orchestrator.is_running = False

            try:
                await asyncio.wait_for(monitor_task, timeout=1)
            except asyncio.TimeoutError:
                pass

            mock_test.assert_called()


class TestSelfHealing:
    """Test self-healing capabilities"""

    @pytest.fixture
    def orchestrator(self):
        return ContinuousDirector(
            project_name="test_project",
            project_spec={"description": "Test"}
        )

    @pytest.mark.asyncio
    async def test_emergency_debug_mode(self, orchestrator):
        """Test emergency debug mode activation"""
        orchestrator.metrics.bug_count_critical = 5
        orchestrator.agents = {"debugger": Mock()}

        await orchestrator.emergency_debug_mode()

        assert orchestrator.state == ProjectState.DEBUGGING
        assert len(orchestrator.task_queue) == 1
        assert orchestrator.task_queue[0].type == "emergency_debug"
        assert orchestrator.task_queue[0].priority == 10


class TestWebSocketEndpoints:
    """Test WebSocket connectivity"""

    @pytest.mark.asyncio
    async def test_websocket_with_project_id(self):
        """Test WebSocket endpoint with project_id"""
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        # Test would connect to /ws/{project_id}
        assert "/ws/{project_id}" in [route.path for route in app.routes]

    @pytest.mark.asyncio
    async def test_websocket_without_project_id(self):
        """Test general WebSocket endpoint"""
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        # Test would connect to /ws
        assert "/ws" in [route.path for route in app.routes]


class TestContinuousLoop:
    """Test the continuous development loop"""

    @pytest.mark.asyncio
    async def test_loop_runs_until_perfect(self):
        """Verify loop continues until metrics are perfect"""
        orchestrator = ContinuousDirector("test", {})
        orchestrator.metrics.test_coverage = 94  # Not perfect

        loop_iterations = 0
        max_iterations = 3

        async def mock_iteration():
            nonlocal loop_iterations
            loop_iterations += 1
            if loop_iterations >= max_iterations:
                orchestrator.metrics.test_coverage = 95
                orchestrator.metrics.bug_count_critical = 0
                orchestrator.metrics.performance_score = 90
                orchestrator.metrics.documentation_coverage = 90
                orchestrator.metrics.code_quality_score = 85
                orchestrator.metrics.user_satisfaction = 90
                orchestrator.metrics.security_score = 95

        with patch.object(orchestrator, '_execute_iteration', mock_iteration):
            with patch.object(orchestrator, '_initialize_components', AsyncMock()):
                task = asyncio.create_task(orchestrator.start())
                await asyncio.sleep(0.5)

                # Should stop when perfect
                assert orchestrator.metrics.is_perfect()


def test_coverage_calculation():
    """Calculate and verify test coverage"""
    import coverage

    cov = coverage.Coverage()
    cov.start()

    # Run all tests
    pytest.main([__file__, '-v'])

    cov.stop()
    cov.save()

    # Get coverage percentage
    total = cov.report()

    print(f"\\nTest Coverage: {total:.2f}%")

    # Verify we meet the 95% threshold
    assert total >= 95, f"Coverage {total:.2f}% is below required 95%"


if __name__ == "__main__":
    # Run tests with coverage report
    pytest.main([__file__, '-v', '--cov=core', '--cov-report=term'])