"""
Integration tests for MyAgent Continuous AI Builder
Tests the full system end-to-end
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestrator.continuous_director import ContinuousDirector, QualityMetrics
from config.settings import settings
from config.database import db_manager


class TestContinuousDirectorIntegration:
    """Test the continuous director with all components"""

    @pytest.mark.asyncio
    async def test_director_initialization(self):
        """Test that director initializes correctly"""
        director = ContinuousDirector(
            project_name="test_project",
            project_spec={"description": "Test project"}
        )

        assert director.project_name == "test_project"
        assert director.state.value == "initializing"
        assert director.iteration_count == 0
        assert isinstance(director.metrics, QualityMetrics)

    @pytest.mark.asyncio
    async def test_component_initialization(self):
        """Test that all components initialize without errors"""
        director = ContinuousDirector(
            project_name="test_init",
            project_spec={"description": "Init test"}
        )

        # This should not raise any exceptions
        await director._initialize_components()

        # Verify components are initialized
        assert hasattr(director, 'project_ledger')
        assert hasattr(director, 'vector_memory')
        assert hasattr(director, 'error_graph')
        assert hasattr(director, 'milestone_tracker')
        assert hasattr(director, 'progress_analyzer')
        assert hasattr(director, 'agents')
        assert len(director.agents) == 6

    @pytest.mark.asyncio
    async def test_quality_metrics(self):
        """Test quality metrics calculations"""
        metrics = QualityMetrics()

        # Initial state should not be perfect
        assert not metrics.is_perfect()

        # Set all metrics to perfect values
        metrics.test_coverage = 95.0
        metrics.bug_count_critical = 0
        metrics.bug_count_minor = 0
        metrics.performance_score = 90.0
        metrics.documentation_coverage = 90.0
        metrics.code_quality_score = 85.0
        metrics.user_satisfaction = 90.0
        metrics.security_score = 95.0

        # Now should be perfect
        assert metrics.is_perfect()

    @pytest.mark.asyncio
    async def test_task_generation(self):
        """Test that tasks are generated correctly"""
        director = ContinuousDirector(
            project_name="test_tasks",
            project_spec={"description": "Task test"}
        )

        # Generate test tasks
        test_tasks = director._generate_test_tasks(priority=2)
        assert isinstance(test_tasks, list)
        assert len(test_tasks) > 0

        # Generate bug fix tasks
        director.metrics.bug_count_critical = 2
        bug_tasks = director._generate_bug_fix_tasks(priority=1)
        assert len(bug_tasks) == 2

    @pytest.mark.asyncio
    async def test_task_prioritization(self):
        """Test task prioritization with dependencies"""
        director = ContinuousDirector(
            project_name="test_priority",
            project_spec={"description": "Priority test"}
        )

        from core.orchestrator.continuous_director import DevelopmentTask, TaskPriority

        # Create tasks with dependencies
        task1 = DevelopmentTask(
            id="task1",
            type="code",
            description="Task 1",
            priority=TaskPriority.NORMAL
        )

        task2 = DevelopmentTask(
            id="task2",
            type="test",
            description="Task 2",
            priority=TaskPriority.HIGH,
            dependencies=["task1"]
        )

        task3 = DevelopmentTask(
            id="task3",
            type="debug",
            description="Task 3",
            priority=TaskPriority.CRITICAL
        )

        tasks = [task2, task1, task3]
        prioritized = director._prioritize_tasks(tasks)

        # Critical task should be first
        assert prioritized[0].id == "task3"
        # Task1 should come before task2 (dependency)
        task1_idx = next(i for i, t in enumerate(prioritized) if t.id == "task1")
        task2_idx = next(i for i, t in enumerate(prioritized) if t.id == "task2")
        assert task1_idx < task2_idx

    @pytest.mark.asyncio
    async def test_checkpoint_creation(self):
        """Test checkpoint creation and loading"""
        director = ContinuousDirector(
            project_name="test_checkpoint",
            project_spec={"description": "Checkpoint test"}
        )

        director.iteration_count = 5
        director.metrics.test_coverage = 50.0

        # Create checkpoint
        await director._create_checkpoint()

        # Verify checkpoint file exists
        checkpoint_path = Path("persistence/snapshots/checkpoint_5.json")
        assert checkpoint_path.exists()

        # Create new director and load state
        director2 = ContinuousDirector(
            project_name="test_checkpoint",
            project_spec={"description": "Checkpoint test"}
        )

        await director2._load_project_state()

        # Should have loaded the iteration count
        assert director2.iteration_count == 5

    @pytest.mark.asyncio
    async def test_state_analysis(self):
        """Test current state analysis"""
        director = ContinuousDirector(
            project_name="test_analysis",
            project_spec={"description": "Analysis test"}
        )

        director.metrics.test_coverage = 80.0
        director.metrics.bug_count_critical = 1

        analysis = await director._analyze_current_state()

        assert "iteration" in analysis
        assert "metrics" in analysis
        assert "weak_metrics" in analysis
        assert analysis["needs_tests"] is True
        assert analysis["has_critical_bugs"] is True


class TestAPIIntegration:
    """Test API endpoints"""

    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test database connection"""
        await db_manager.connect()
        is_healthy = await db_manager.health_check()
        assert is_healthy
        await db_manager.disconnect()


class TestMemorySystemsIntegration:
    """Test memory systems working together"""

    @pytest.mark.asyncio
    async def test_memory_initialization(self):
        """Test that memory systems can be initialized"""
        from core.memory.project_ledger import ProjectLedger
        from core.memory.vector_memory import VectorMemory
        from core.memory.error_knowledge_graph import ErrorKnowledgeGraph

        ledger = ProjectLedger("test_memory")
        vector_mem = VectorMemory("test_memory")
        error_graph = ErrorKnowledgeGraph()

        assert ledger is not None
        assert vector_mem is not None
        assert error_graph is not None


@pytest.mark.asyncio
async def test_full_system_smoke_test():
    """Smoke test - verify system can start without crashing"""
    director = ContinuousDirector(
        project_name="smoke_test",
        project_spec={
            "description": "Smoke test project",
            "requirements": ["Create a hello world app"]
        }
    )

    # Initialize components
    await director._initialize_components()

    # Verify all agents are initialized
    assert len(director.agents) == 6
    assert "coder" in director.agents
    assert "tester" in director.agents
    assert "debugger" in director.agents
    assert "architect" in director.agents
    assert "analyzer" in director.agents
    assert "ui_refiner" in director.agents

    # Verify metrics start at zero/defaults
    assert director.metrics.test_coverage == 0.0
    assert not director.metrics.is_perfect()

    print("✅ Full system smoke test passed!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
