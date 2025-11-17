#!/usr/bin/env python3
"""
Multi-Agent Coordination - Comprehensive Integration Tests

Tests the coordination and communication between multiple agents working together
on complex development tasks. Validates the orchestration system's ability to
manage concurrent agents, handle task dependencies, and maintain system coherence.

Testing methodologies applied:
- Integration testing for agent coordination workflows
- BDD scenarios for multi-agent collaboration patterns
- Message-passing validation for agent communication
- Concurrency testing for parallel agent operations
- Workflow testing for complex task orchestration
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

# Import test fixtures
from tests.fixtures.test_data import TEST_DATA
from tests.fixtures.agent_fixtures import (
    MockCoderAgent,
    MockTesterAgent,
    MockDebuggerAgent,
    MockArchitectAgent,
    MockAnalyzerAgent,
    MockUIRefinerAgent
)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class AgentRole(Enum):
    """Agent roles in the system"""
    CODER = "coder"
    TESTER = "tester"
    DEBUGGER = "debugger"
    ARCHITECT = "architect"
    ANALYZER = "analyzer"
    UI_REFINER = "ui_refiner"


@dataclass
class AgentTask:
    """Task assigned to an agent"""
    task_id: str
    task_type: str
    description: str
    assigned_agent: AgentRole
    priority: int
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class AgentMessage:
    """Message passed between agents"""
    message_id: str
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


@dataclass
class WorkflowStep:
    """Step in a multi-agent workflow"""
    step_id: str
    description: str
    required_agents: List[AgentRole]
    tasks: List[AgentTask]
    dependencies: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    timeout_seconds: int = 300


class MockOrchestrator:
    """Mock orchestrator for multi-agent coordination testing"""

    def __init__(self):
        self.agents: Dict[AgentRole, Any] = {}
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, AgentTask] = {}
        self.message_queue: List[AgentMessage] = []
        self.message_history: List[AgentMessage] = []
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.coordination_metrics: Dict[str, Any] = {
            "messages_sent": 0,
            "tasks_completed": 0,
            "coordination_errors": 0,
            "average_response_time": 0.0
        }

    async def initialize(self):
        """Initialize orchestrator with agents"""
        # Create agent instances
        self.agents[AgentRole.CODER] = MockCoderAgent()
        self.agents[AgentRole.TESTER] = MockTesterAgent()
        self.agents[AgentRole.DEBUGGER] = MockDebuggerAgent()
        self.agents[AgentRole.ARCHITECT] = MockArchitectAgent()
        self.agents[AgentRole.ANALYZER] = MockAnalyzerAgent()
        self.agents[AgentRole.UI_REFINER] = MockUIRefinerAgent()

        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize()

    async def assign_task(self, task: AgentTask) -> str:
        """Assign task to appropriate agent"""
        if task.assigned_agent not in self.agents:
            raise ValueError(f"Agent {task.assigned_agent} not available")

        # Check dependencies
        if not await self._check_task_dependencies(task):
            task.status = TaskStatus.BLOCKED
            return task.task_id

        self.active_tasks[task.task_id] = task
        return task.task_id

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a single task"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.active_tasks[task_id]
        agent = self.agents[task.assigned_agent]

        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()

            # Execute task with agent
            result = await agent.execute_task(task.task_type, task.parameters)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]

            self.coordination_metrics["tasks_completed"] += 1

            return {
                "task_id": task_id,
                "status": task.status.value,
                "result": result,
                "execution_time": (task.completed_at - task.started_at).total_seconds()
            }

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()

            self.coordination_metrics["coordination_errors"] += 1

            return {
                "task_id": task_id,
                "status": task.status.value,
                "error": str(e)
            }

    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute multi-step workflow with multiple agents"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow_steps = self.workflows[workflow_id]
        workflow_results = []
        workflow_start = datetime.now()

        try:
            for step in workflow_steps:
                # Check step dependencies
                if not await self._check_step_dependencies(step, workflow_results):
                    raise ValueError(f"Step {step.step_id} dependencies not satisfied")

                step_start = datetime.now()
                step_results = []

                if step.parallel_execution:
                    # Execute tasks in parallel
                    tasks = [self.execute_task(task.task_id) for task in step.tasks]
                    step_results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Execute tasks sequentially
                    for task in step.tasks:
                        result = await self.execute_task(task.task_id)
                        step_results.append(result)

                step_duration = (datetime.now() - step_start).total_seconds()

                step_result = {
                    "step_id": step.step_id,
                    "description": step.description,
                    "agents_involved": [agent.value for agent in step.required_agents],
                    "task_results": step_results,
                    "duration": step_duration,
                    "status": "completed"
                }

                workflow_results.append(step_result)

                # Process any inter-agent messages generated during step
                await self._process_message_queue()

            workflow_duration = (datetime.now() - workflow_start).total_seconds()

            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "duration": workflow_duration,
                "steps": workflow_results,
                "coordination_metrics": self.coordination_metrics.copy()
            }

        except Exception as e:
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "partial_results": workflow_results
            }

    async def send_message(self, message: AgentMessage) -> bool:
        """Send message between agents"""
        if message.from_agent not in self.agents or message.to_agent not in self.agents:
            return False

        # Add to message queue for processing
        self.message_queue.append(message)
        self.message_history.append(message)
        self.coordination_metrics["messages_sent"] += 1

        return True

    async def coordinate_agents(self, coordination_scenario: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents for specific scenarios"""
        coordination_start = datetime.now()
        coordination_result = {
            "scenario": coordination_scenario,
            "participants": [],
            "messages_exchanged": [],
            "outcome": "unknown"
        }

        try:
            if coordination_scenario == "code_review_process":
                await self._coordinate_code_review(parameters, coordination_result)

            elif coordination_scenario == "bug_fixing_workflow":
                await self._coordinate_bug_fixing(parameters, coordination_result)

            elif coordination_scenario == "feature_development":
                await self._coordinate_feature_development(parameters, coordination_result)

            elif coordination_scenario == "performance_optimization":
                await self._coordinate_performance_optimization(parameters, coordination_result)

            elif coordination_scenario == "ui_improvement":
                await self._coordinate_ui_improvement(parameters, coordination_result)

            coordination_result["duration"] = (datetime.now() - coordination_start).total_seconds()
            coordination_result["outcome"] = "success"

        except Exception as e:
            coordination_result["outcome"] = "failed"
            coordination_result["error"] = str(e)

        return coordination_result

    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination performance metrics"""
        total_tasks = len(self.completed_tasks) + len(self.active_tasks)
        completion_rate = len(self.completed_tasks) / total_tasks if total_tasks > 0 else 0

        # Calculate average response time
        response_times = []
        for task in self.completed_tasks.values():
            if task.started_at and task.completed_at:
                response_times.append((task.completed_at - task.started_at).total_seconds())

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return {
            **self.coordination_metrics,
            "average_response_time": avg_response_time,
            "task_completion_rate": completion_rate,
            "active_agents": len(self.agents),
            "message_queue_size": len(self.message_queue),
            "total_workflow_steps": sum(len(steps) for steps in self.workflows.values())
        }

    # Helper methods for coordination scenarios

    async def _coordinate_code_review(self, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Coordinate code review process between agents"""
        # Coder presents code for review
        coder = self.agents[AgentRole.CODER]
        architect = self.agents[AgentRole.ARCHITECT]
        tester = self.agents[AgentRole.TESTER]

        participants = [AgentRole.CODER, AgentRole.ARCHITECT, AgentRole.TESTER]
        result["participants"] = [p.value for p in participants]

        # Architect reviews architecture
        arch_review = await architect.execute_task("review_architecture", parameters)

        # Tester reviews testability
        test_review = await tester.execute_task("review_testability", parameters)

        # Simulate message exchange
        messages = [
            AgentMessage(
                message_id="review_1",
                from_agent=AgentRole.ARCHITECT,
                to_agent=AgentRole.CODER,
                message_type="review_feedback",
                content=arch_review
            ),
            AgentMessage(
                message_id="review_2",
                from_agent=AgentRole.TESTER,
                to_agent=AgentRole.CODER,
                message_type="test_feedback",
                content=test_review
            )
        ]

        for message in messages:
            await self.send_message(message)

        result["messages_exchanged"] = [m.message_type for m in messages]

    async def _coordinate_bug_fixing(self, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Coordinate bug fixing workflow"""
        debugger = self.agents[AgentRole.DEBUGGER]
        coder = self.agents[AgentRole.CODER]
        tester = self.agents[AgentRole.TESTER]

        participants = [AgentRole.DEBUGGER, AgentRole.CODER, AgentRole.TESTER]
        result["participants"] = [p.value for p in participants]

        # Debugger analyzes the bug
        debug_analysis = await debugger.execute_task("analyze_bug", parameters)

        # Coder implements fix based on analysis
        fix_implementation = await coder.execute_task("implement_fix", debug_analysis)

        # Tester verifies the fix
        verification = await tester.execute_task("verify_fix", fix_implementation)

        # Simulate coordination messages
        messages = [
            AgentMessage(
                message_id="bug_analysis",
                from_agent=AgentRole.DEBUGGER,
                to_agent=AgentRole.CODER,
                message_type="analysis_report",
                content=debug_analysis
            ),
            AgentMessage(
                message_id="fix_ready",
                from_agent=AgentRole.CODER,
                to_agent=AgentRole.TESTER,
                message_type="fix_implementation",
                content=fix_implementation
            ),
            AgentMessage(
                message_id="verification_result",
                from_agent=AgentRole.TESTER,
                to_agent=AgentRole.DEBUGGER,
                message_type="verification_report",
                content=verification
            )
        ]

        for message in messages:
            await self.send_message(message)

        result["messages_exchanged"] = [m.message_type for m in messages]

    async def _coordinate_feature_development(self, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Coordinate feature development across agents"""
        architect = self.agents[AgentRole.ARCHITECT]
        coder = self.agents[AgentRole.CODER]
        tester = self.agents[AgentRole.TESTER]
        ui_refiner = self.agents[AgentRole.UI_REFINER]

        participants = [AgentRole.ARCHITECT, AgentRole.CODER, AgentRole.TESTER, AgentRole.UI_REFINER]
        result["participants"] = [p.value for p in participants]

        # Architect designs the feature
        design = await architect.execute_task("design_feature", parameters)

        # Coder implements the feature
        implementation = await coder.execute_task("implement_feature", design)

        # UI Refiner improves the interface
        ui_improvements = await ui_refiner.execute_task("improve_ui", implementation)

        # Tester creates and runs tests
        test_results = await tester.execute_task("test_feature", ui_improvements)

        # Coordinate through messages
        messages = [
            AgentMessage(
                message_id="design_complete",
                from_agent=AgentRole.ARCHITECT,
                to_agent=AgentRole.CODER,
                message_type="design_specification",
                content=design
            ),
            AgentMessage(
                message_id="implementation_ready",
                from_agent=AgentRole.CODER,
                to_agent=AgentRole.UI_REFINER,
                message_type="feature_implementation",
                content=implementation
            ),
            AgentMessage(
                message_id="ui_ready",
                from_agent=AgentRole.UI_REFINER,
                to_agent=AgentRole.TESTER,
                message_type="ui_complete",
                content=ui_improvements
            )
        ]

        for message in messages:
            await self.send_message(message)

        result["messages_exchanged"] = [m.message_type for m in messages]

    async def _coordinate_performance_optimization(self, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Coordinate performance optimization workflow"""
        analyzer = self.agents[AgentRole.ANALYZER]
        coder = self.agents[AgentRole.CODER]
        tester = self.agents[AgentRole.TESTER]

        participants = [AgentRole.ANALYZER, AgentRole.CODER, AgentRole.TESTER]
        result["participants"] = [p.value for p in participants]

        # Analyzer identifies performance bottlenecks
        analysis = await analyzer.execute_task("analyze_performance", parameters)

        # Coder implements optimizations
        optimizations = await coder.execute_task("implement_optimizations", analysis)

        # Tester validates performance improvements
        validation = await tester.execute_task("validate_performance", optimizations)

        # Exchange optimization insights
        messages = [
            AgentMessage(
                message_id="perf_analysis",
                from_agent=AgentRole.ANALYZER,
                to_agent=AgentRole.CODER,
                message_type="performance_report",
                content=analysis
            ),
            AgentMessage(
                message_id="optimizations_done",
                from_agent=AgentRole.CODER,
                to_agent=AgentRole.TESTER,
                message_type="optimization_implementation",
                content=optimizations
            )
        ]

        for message in messages:
            await self.send_message(message)

        result["messages_exchanged"] = [m.message_type for m in messages]

    async def _coordinate_ui_improvement(self, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Coordinate UI improvement workflow"""
        ui_refiner = self.agents[AgentRole.UI_REFINER]
        coder = self.agents[AgentRole.CODER]
        tester = self.agents[AgentRole.TESTER]

        participants = [AgentRole.UI_REFINER, AgentRole.CODER, AgentRole.TESTER]
        result["participants"] = [p.value for p in participants]

        # UI Refiner analyzes current UI
        ui_analysis = await ui_refiner.execute_task("analyze_ui", parameters)

        # UI Refiner designs improvements
        improvements = await ui_refiner.execute_task("design_improvements", ui_analysis)

        # Coder implements UI changes
        implementation = await coder.execute_task("implement_ui_changes", improvements)

        # Tester performs UI testing
        ui_tests = await tester.execute_task("test_ui", implementation)

        # Coordinate UI improvements
        messages = [
            AgentMessage(
                message_id="ui_analysis_complete",
                from_agent=AgentRole.UI_REFINER,
                to_agent=AgentRole.CODER,
                message_type="ui_improvement_plan",
                content=improvements
            ),
            AgentMessage(
                message_id="ui_implementation_done",
                from_agent=AgentRole.CODER,
                to_agent=AgentRole.TESTER,
                message_type="ui_changes_ready",
                content=implementation
            )
        ]

        for message in messages:
            await self.send_message(message)

        result["messages_exchanged"] = [m.message_type for m in messages]

    async def _check_task_dependencies(self, task: AgentTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dependency_id in task.dependencies:
            if dependency_id not in self.completed_tasks:
                return False
        return True

    async def _check_step_dependencies(self, step: WorkflowStep, workflow_results: List[Dict[str, Any]]) -> bool:
        """Check if workflow step dependencies are satisfied"""
        completed_steps = {result["step_id"] for result in workflow_results}
        return all(dep in completed_steps for dep in step.dependencies)

    async def _process_message_queue(self):
        """Process queued inter-agent messages"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            # In a real system, this would deliver the message to the target agent
            # For testing, we just track that it was processed
            pass


@pytest.fixture
def orchestrator():
    """Fixture providing mock orchestrator"""
    return MockOrchestrator()


@pytest.fixture
def sample_tasks():
    """Fixture providing sample tasks for testing"""
    return [
        AgentTask(
            task_id="task_001",
            task_type="generate_function",
            description="Generate a fibonacci function",
            assigned_agent=AgentRole.CODER,
            priority=5,
            parameters={"function_name": "fibonacci", "language": "python"}
        ),
        AgentTask(
            task_id="task_002",
            task_type="create_tests",
            description="Create unit tests for fibonacci function",
            assigned_agent=AgentRole.TESTER,
            priority=4,
            dependencies=["task_001"],
            parameters={"target_function": "fibonacci", "coverage_target": 95}
        ),
        AgentTask(
            task_id="task_003",
            task_type="analyze_performance",
            description="Analyze fibonacci function performance",
            assigned_agent=AgentRole.ANALYZER,
            priority=3,
            dependencies=["task_001"],
            parameters={"function_name": "fibonacci", "analysis_type": "complexity"}
        )
    ]


@pytest.fixture
def sample_workflow():
    """Fixture providing sample multi-agent workflow"""
    # Create tasks for the workflow
    tasks = [
        AgentTask(
            task_id="wf_task_001",
            task_type="design_feature",
            description="Design user authentication feature",
            assigned_agent=AgentRole.ARCHITECT,
            priority=8
        ),
        AgentTask(
            task_id="wf_task_002",
            task_type="implement_feature",
            description="Implement authentication feature",
            assigned_agent=AgentRole.CODER,
            priority=7
        ),
        AgentTask(
            task_id="wf_task_003",
            task_type="test_feature",
            description="Test authentication feature",
            assigned_agent=AgentRole.TESTER,
            priority=6
        ),
        AgentTask(
            task_id="wf_task_004",
            task_type="analyze_security",
            description="Analyze authentication security",
            assigned_agent=AgentRole.ANALYZER,
            priority=7
        )
    ]

    steps = [
        WorkflowStep(
            step_id="design_phase",
            description="Design the authentication feature",
            required_agents=[AgentRole.ARCHITECT],
            tasks=[tasks[0]],
            parallel_execution=False
        ),
        WorkflowStep(
            step_id="implementation_phase",
            description="Implement and analyze the feature",
            required_agents=[AgentRole.CODER, AgentRole.ANALYZER],
            tasks=[tasks[1], tasks[3]],
            dependencies=["design_phase"],
            parallel_execution=True
        ),
        WorkflowStep(
            step_id="testing_phase",
            description="Test the implemented feature",
            required_agents=[AgentRole.TESTER],
            tasks=[tasks[2]],
            dependencies=["implementation_phase"],
            parallel_execution=False
        )
    ]

    return "auth_feature_workflow", steps, tasks


class TestMultiAgentCoordination:
    """Comprehensive tests for multi-agent coordination"""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization with agents"""
        await orchestrator.initialize()

        # All agents should be initialized
        expected_agents = [AgentRole.CODER, AgentRole.TESTER, AgentRole.DEBUGGER,
                          AgentRole.ARCHITECT, AgentRole.ANALYZER, AgentRole.UI_REFINER]

        assert len(orchestrator.agents) == len(expected_agents)
        for agent_role in expected_agents:
            assert agent_role in orchestrator.agents
            # Verify agent has been initialized (mock agents have name attribute after init)
            assert hasattr(orchestrator.agents[agent_role], 'name')

    @pytest.mark.asyncio
    async def test_single_task_assignment_and_execution(self, orchestrator, sample_tasks):
        """Test assignment and execution of single tasks"""
        await orchestrator.initialize()

        task = sample_tasks[0]  # Coder task with no dependencies

        # Assign task
        task_id = await orchestrator.assign_task(task)
        assert task_id == task.task_id
        assert task_id in orchestrator.active_tasks

        # Execute task
        result = await orchestrator.execute_task(task_id)

        assert result["task_id"] == task_id
        assert result["status"] == TaskStatus.COMPLETED.value
        assert "result" in result
        assert "execution_time" in result

        # Task should be moved to completed
        assert task_id in orchestrator.completed_tasks
        assert task_id not in orchestrator.active_tasks

    @pytest.mark.asyncio
    async def test_task_dependency_handling(self, orchestrator, sample_tasks):
        """Test handling of task dependencies"""
        await orchestrator.initialize()

        dependent_task = sample_tasks[1]  # Tester task that depends on task_001

        # Try to assign dependent task before dependency is completed
        task_id = await orchestrator.assign_task(dependent_task)
        assert dependent_task.status == TaskStatus.BLOCKED

        # Complete the dependency first
        dependency_task = sample_tasks[0]
        await orchestrator.assign_task(dependency_task)
        await orchestrator.execute_task(dependency_task.task_id)

        # Now the dependent task should be executable
        dependent_task.status = TaskStatus.PENDING  # Reset status
        task_id = await orchestrator.assign_task(dependent_task)
        assert dependent_task.status != TaskStatus.BLOCKED

        # Execute dependent task
        result = await orchestrator.execute_task(task_id)
        assert result["status"] == TaskStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_inter_agent_messaging(self, orchestrator):
        """Test message passing between agents"""
        await orchestrator.initialize()

        # Create a message from coder to tester
        message = AgentMessage(
            message_id="msg_001",
            from_agent=AgentRole.CODER,
            to_agent=AgentRole.TESTER,
            message_type="code_ready_for_testing",
            content={"code": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"},
            correlation_id="task_001"
        )

        # Send message
        success = await orchestrator.send_message(message)
        assert success is True

        # Verify message tracking
        assert len(orchestrator.message_history) == 1
        assert orchestrator.message_history[0].message_id == "msg_001"
        assert orchestrator.coordination_metrics["messages_sent"] == 1

    @pytest.mark.asyncio
    async def test_parallel_task_execution(self, orchestrator):
        """Test parallel execution of independent tasks"""
        await orchestrator.initialize()

        # Create independent tasks for different agents
        tasks = [
            AgentTask(
                task_id="parallel_001",
                task_type="analyze_code",
                description="Analyze code quality",
                assigned_agent=AgentRole.ANALYZER,
                priority=5
            ),
            AgentTask(
                task_id="parallel_002",
                task_type="design_architecture",
                description="Design system architecture",
                assigned_agent=AgentRole.ARCHITECT,
                priority=5
            ),
            AgentTask(
                task_id="parallel_003",
                task_type="improve_ui",
                description="Improve user interface",
                assigned_agent=AgentRole.UI_REFINER,
                priority=5
            )
        ]

        # Assign all tasks
        task_ids = []
        for task in tasks:
            task_id = await orchestrator.assign_task(task)
            task_ids.append(task_id)

        # Execute tasks in parallel
        start_time = time.time()
        execution_tasks = [orchestrator.execute_task(tid) for tid in task_ids]
        results = await asyncio.gather(*execution_tasks)
        execution_time = time.time() - start_time

        # All tasks should complete successfully
        assert len(results) == 3
        for result in results:
            assert result["status"] == TaskStatus.COMPLETED.value

        # Parallel execution should be faster than sequential
        # (Mock agents have small delays, so this tests concurrency)
        assert execution_time < 1.0  # Should complete quickly in parallel

    @pytest.mark.asyncio
    async def test_workflow_execution(self, orchestrator, sample_workflow):
        """Test execution of multi-step workflow"""
        await orchestrator.initialize()

        workflow_id, steps, tasks = sample_workflow

        # Register workflow
        orchestrator.workflows[workflow_id] = steps

        # Assign all workflow tasks
        for task in tasks:
            await orchestrator.assign_task(task)

        # Execute workflow
        result = await orchestrator.execute_workflow(workflow_id)

        assert result["workflow_id"] == workflow_id
        assert result["status"] == "completed"
        assert "duration" in result
        assert len(result["steps"]) == 3

        # Verify step execution order
        step_ids = [step["step_id"] for step in result["steps"]]
        assert step_ids == ["design_phase", "implementation_phase", "testing_phase"]

        # Verify parallel execution in implementation phase
        impl_step = result["steps"][1]
        assert len(impl_step["task_results"]) == 2  # Two parallel tasks
        assert AgentRole.CODER.value in impl_step["agents_involved"]
        assert AgentRole.ANALYZER.value in impl_step["agents_involved"]

    @pytest.mark.asyncio
    async def test_code_review_coordination(self, orchestrator):
        """Test code review coordination scenario"""
        await orchestrator.initialize()

        parameters = {
            "code": "def calculate_sum(numbers): return sum(numbers)",
            "file_path": "utils/calculations.py",
            "review_type": "quality_and_architecture"
        }

        result = await orchestrator.coordinate_agents("code_review_process", parameters)

        assert result["scenario"] == "code_review_process"
        assert result["outcome"] == "success"
        assert len(result["participants"]) == 3
        assert AgentRole.CODER.value in result["participants"]
        assert AgentRole.ARCHITECT.value in result["participants"]
        assert AgentRole.TESTER.value in result["participants"]
        assert len(result["messages_exchanged"]) > 0

        # Should have coordination messages
        assert "review_feedback" in result["messages_exchanged"] or "test_feedback" in result["messages_exchanged"]

    @pytest.mark.asyncio
    async def test_bug_fixing_coordination(self, orchestrator):
        """Test bug fixing coordination scenario"""
        await orchestrator.initialize()

        parameters = {
            "bug_report": "Division by zero error in calculation function",
            "error_trace": "ZeroDivisionError at line 42",
            "severity": "high"
        }

        result = await orchestrator.coordinate_agents("bug_fixing_workflow", parameters)

        assert result["scenario"] == "bug_fixing_workflow"
        assert result["outcome"] == "success"
        assert len(result["participants"]) == 3
        assert AgentRole.DEBUGGER.value in result["participants"]
        assert AgentRole.CODER.value in result["participants"]
        assert AgentRole.TESTER.value in result["participants"]

        # Should have specific coordination messages for bug fixing
        expected_messages = ["analysis_report", "fix_implementation", "verification_report"]
        assert any(msg in result["messages_exchanged"] for msg in expected_messages)

    @pytest.mark.asyncio
    async def test_feature_development_coordination(self, orchestrator):
        """Test feature development coordination scenario"""
        await orchestrator.initialize()

        parameters = {
            "feature_requirements": "User profile management system",
            "complexity": "medium",
            "target_users": "end_users"
        }

        result = await orchestrator.coordinate_agents("feature_development", parameters)

        assert result["scenario"] == "feature_development"
        assert result["outcome"] == "success"
        assert len(result["participants"]) == 4  # All agents involved
        assert AgentRole.ARCHITECT.value in result["participants"]
        assert AgentRole.CODER.value in result["participants"]
        assert AgentRole.UI_REFINER.value in result["participants"]
        assert AgentRole.TESTER.value in result["participants"]

        # Should coordinate through multiple phases
        expected_message_types = ["design_specification", "feature_implementation", "ui_complete"]
        assert any(msg in result["messages_exchanged"] for msg in expected_message_types)

    @pytest.mark.asyncio
    async def test_performance_optimization_coordination(self, orchestrator):
        """Test performance optimization coordination"""
        await orchestrator.initialize()

        parameters = {
            "performance_issue": "Slow database queries",
            "target_improvement": "50% response time reduction",
            "affected_endpoints": ["/api/users", "/api/data"]
        }

        result = await orchestrator.coordinate_agents("performance_optimization", parameters)

        assert result["scenario"] == "performance_optimization"
        assert result["outcome"] == "success"
        assert AgentRole.ANALYZER.value in result["participants"]
        assert AgentRole.CODER.value in result["participants"]
        assert AgentRole.TESTER.value in result["participants"]

        # Should have performance-specific coordination
        assert "performance_report" in result["messages_exchanged"] or "optimization_implementation" in result["messages_exchanged"]

    @pytest.mark.asyncio
    async def test_ui_improvement_coordination(self, orchestrator):
        """Test UI improvement coordination"""
        await orchestrator.initialize()

        parameters = {
            "ui_issues": ["Low contrast", "Poor mobile responsiveness", "Accessibility problems"],
            "target_users": "all_users",
            "priority_level": "high"
        }

        result = await orchestrator.coordinate_agents("ui_improvement", parameters)

        assert result["scenario"] == "ui_improvement"
        assert result["outcome"] == "success"
        assert AgentRole.UI_REFINER.value in result["participants"]
        assert AgentRole.CODER.value in result["participants"]
        assert AgentRole.TESTER.value in result["participants"]

        # Should have UI-specific coordination messages
        assert "ui_improvement_plan" in result["messages_exchanged"] or "ui_changes_ready" in result["messages_exchanged"]

    @pytest.mark.asyncio
    async def test_coordination_metrics_tracking(self, orchestrator, sample_tasks):
        """Test coordination metrics tracking"""
        await orchestrator.initialize()

        # Execute some tasks to generate metrics
        for task in sample_tasks[:2]:  # Execute first two tasks
            if not task.dependencies:  # Skip dependent tasks for simplicity
                await orchestrator.assign_task(task)
                await orchestrator.execute_task(task.task_id)

        # Send some messages
        message = AgentMessage(
            message_id="metrics_test",
            from_agent=AgentRole.CODER,
            to_agent=AgentRole.TESTER,
            message_type="test_message",
            content={}
        )
        await orchestrator.send_message(message)

        # Get metrics
        metrics = await orchestrator.get_coordination_metrics()

        assert "messages_sent" in metrics
        assert "tasks_completed" in metrics
        assert "coordination_errors" in metrics
        assert "average_response_time" in metrics
        assert "task_completion_rate" in metrics
        assert "active_agents" in metrics

        assert metrics["messages_sent"] > 0
        assert metrics["tasks_completed"] > 0
        assert metrics["active_agents"] == 6  # All agent types
        assert 0.0 <= metrics["task_completion_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_error_handling_in_coordination(self, orchestrator):
        """Test error handling during coordination"""
        await orchestrator.initialize()

        # Create a task for non-existent agent
        invalid_task = AgentTask(
            task_id="invalid_task",
            task_type="invalid_operation",
            description="Invalid task",
            assigned_agent=AgentRole.CODER,  # Valid agent but will cause execution error
            priority=5,
            parameters={"invalid_param": "error_trigger"}
        )

        # Assign and execute task (should handle error gracefully)
        task_id = await orchestrator.assign_task(invalid_task)
        result = await orchestrator.execute_task(task_id)

        assert result["status"] == TaskStatus.FAILED.value
        assert "error" in result
        assert orchestrator.coordination_metrics["coordination_errors"] > 0

    @pytest.mark.asyncio
    async def test_message_correlation(self, orchestrator):
        """Test message correlation for tracking related communications"""
        await orchestrator.initialize()

        correlation_id = "feature_dev_001"

        # Send related messages with same correlation ID
        messages = [
            AgentMessage(
                message_id="msg_001",
                from_agent=AgentRole.ARCHITECT,
                to_agent=AgentRole.CODER,
                message_type="design_spec",
                content={"spec": "authentication_design"},
                correlation_id=correlation_id
            ),
            AgentMessage(
                message_id="msg_002",
                from_agent=AgentRole.CODER,
                to_agent=AgentRole.TESTER,
                message_type="implementation_ready",
                content={"code": "auth_implementation"},
                correlation_id=correlation_id
            )
        ]

        for message in messages:
            await orchestrator.send_message(message)

        # Find related messages by correlation ID
        related_messages = [
            msg for msg in orchestrator.message_history
            if msg.correlation_id == correlation_id
        ]

        assert len(related_messages) == 2
        assert all(msg.correlation_id == correlation_id for msg in related_messages)

    @pytest.mark.asyncio
    async def test_concurrent_workflows(self, orchestrator):
        """Test execution of concurrent workflows"""
        await orchestrator.initialize()

        # Create two independent workflows
        workflow1_tasks = [
            AgentTask(
                task_id="wf1_task1",
                task_type="analyze_data",
                description="Analyze user data",
                assigned_agent=AgentRole.ANALYZER,
                priority=5
            )
        ]

        workflow2_tasks = [
            AgentTask(
                task_id="wf2_task1",
                task_type="improve_ui",
                description="Improve dashboard UI",
                assigned_agent=AgentRole.UI_REFINER,
                priority=5
            )
        ]

        workflow1_steps = [
            WorkflowStep(
                step_id="analyze_step",
                description="Data analysis step",
                required_agents=[AgentRole.ANALYZER],
                tasks=workflow1_tasks
            )
        ]

        workflow2_steps = [
            WorkflowStep(
                step_id="ui_step",
                description="UI improvement step",
                required_agents=[AgentRole.UI_REFINER],
                tasks=workflow2_tasks
            )
        ]

        # Register workflows
        orchestrator.workflows["data_analysis_workflow"] = workflow1_steps
        orchestrator.workflows["ui_improvement_workflow"] = workflow2_steps

        # Assign tasks
        for task in workflow1_tasks + workflow2_tasks:
            await orchestrator.assign_task(task)

        # Execute workflows concurrently
        workflow_tasks = [
            orchestrator.execute_workflow("data_analysis_workflow"),
            orchestrator.execute_workflow("ui_improvement_workflow")
        ]

        results = await asyncio.gather(*workflow_tasks)

        # Both workflows should complete successfully
        assert len(results) == 2
        assert all(result["status"] == "completed" for result in results)
        assert results[0]["workflow_id"] == "data_analysis_workflow"
        assert results[1]["workflow_id"] == "ui_improvement_workflow"

    @pytest.mark.asyncio
    async def test_workflow_step_dependencies(self, orchestrator):
        """Test workflow steps with dependencies execute in correct order"""
        await orchestrator.initialize()

        # Create workflow with strict step dependencies
        tasks = [
            AgentTask(
                task_id="step1_task",
                task_type="design_system",
                description="Design system architecture",
                assigned_agent=AgentRole.ARCHITECT,
                priority=8
            ),
            AgentTask(
                task_id="step2_task",
                task_type="implement_core",
                description="Implement core functionality",
                assigned_agent=AgentRole.CODER,
                priority=7
            ),
            AgentTask(
                task_id="step3_task",
                task_type="test_system",
                description="Test the system",
                assigned_agent=AgentRole.TESTER,
                priority=6
            )
        ]

        steps = [
            WorkflowStep(
                step_id="design",
                description="System design",
                required_agents=[AgentRole.ARCHITECT],
                tasks=[tasks[0]]
            ),
            WorkflowStep(
                step_id="implementation",
                description="Core implementation",
                required_agents=[AgentRole.CODER],
                tasks=[tasks[1]],
                dependencies=["design"]  # Depends on design step
            ),
            WorkflowStep(
                step_id="testing",
                description="System testing",
                required_agents=[AgentRole.TESTER],
                tasks=[tasks[2]],
                dependencies=["implementation"]  # Depends on implementation step
            )
        ]

        # Register workflow
        orchestrator.workflows["sequential_workflow"] = steps

        # Assign tasks
        for task in tasks:
            await orchestrator.assign_task(task)

        # Execute workflow
        result = await orchestrator.execute_workflow("sequential_workflow")

        assert result["status"] == "completed"
        assert len(result["steps"]) == 3

        # Verify execution order
        step_ids = [step["step_id"] for step in result["steps"]]
        assert step_ids == ["design", "implementation", "testing"]


class TestAgentCommunicationPatterns:
    """Tests for various agent communication patterns"""

    @pytest.mark.asyncio
    async def test_broadcast_communication(self, orchestrator):
        """Test broadcast communication from one agent to multiple agents"""
        await orchestrator.initialize()

        # Simulate architect broadcasting design to all development agents
        broadcast_messages = [
            AgentMessage(
                message_id=f"broadcast_{i}",
                from_agent=AgentRole.ARCHITECT,
                to_agent=agent,
                message_type="design_broadcast",
                content={"design": "new_system_architecture", "version": "1.0"},
                correlation_id="architecture_update"
            )
            for i, agent in enumerate([AgentRole.CODER, AgentRole.TESTER, AgentRole.UI_REFINER, AgentRole.ANALYZER])
        ]

        # Send all broadcast messages
        for message in broadcast_messages:
            success = await orchestrator.send_message(message)
            assert success is True

        # Verify all messages were sent
        assert len(orchestrator.message_history) == 4
        assert orchestrator.coordination_metrics["messages_sent"] == 4

        # All messages should have same correlation ID
        correlation_ids = [msg.correlation_id for msg in orchestrator.message_history]
        assert all(cid == "architecture_update" for cid in correlation_ids)

    @pytest.mark.asyncio
    async def test_request_response_pattern(self, orchestrator):
        """Test request-response communication pattern"""
        await orchestrator.initialize()

        # Coder requests review from architect
        request = AgentMessage(
            message_id="review_request",
            from_agent=AgentRole.CODER,
            to_agent=AgentRole.ARCHITECT,
            message_type="review_request",
            content={"code": "new_implementation", "priority": "high"},
            correlation_id="code_review_session"
        )

        # Architect responds with review
        response = AgentMessage(
            message_id="review_response",
            from_agent=AgentRole.ARCHITECT,
            to_agent=AgentRole.CODER,
            message_type="review_response",
            content={"status": "approved", "suggestions": ["optimize_loops", "add_comments"]},
            correlation_id="code_review_session"
        )

        # Send request and response
        await orchestrator.send_message(request)
        await orchestrator.send_message(response)

        # Verify request-response pair
        session_messages = [
            msg for msg in orchestrator.message_history
            if msg.correlation_id == "code_review_session"
        ]

        assert len(session_messages) == 2
        assert session_messages[0].message_type == "review_request"
        assert session_messages[1].message_type == "review_response"

    @pytest.mark.asyncio
    async def test_pipeline_communication(self, orchestrator):
        """Test pipeline communication pattern"""
        await orchestrator.initialize()

        # Create a processing pipeline: Architect -> Coder -> Tester -> Analyzer
        pipeline_steps = [
            ("design", AgentRole.ARCHITECT, AgentRole.CODER),
            ("implementation", AgentRole.CODER, AgentRole.TESTER),
            ("testing", AgentRole.TESTER, AgentRole.ANALYZER),
        ]

        correlation_id = "feature_pipeline"
        content = {"feature": "user_authentication", "stage": "initial"}

        # Send messages through pipeline
        for i, (stage, from_agent, to_agent) in enumerate(pipeline_steps):
            message = AgentMessage(
                message_id=f"pipeline_step_{i}",
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=f"{stage}_handoff",
                content={**content, "stage": stage, "step": i},
                correlation_id=correlation_id
            )

            await orchestrator.send_message(message)

        # Verify pipeline flow
        pipeline_messages = [
            msg for msg in orchestrator.message_history
            if msg.correlation_id == correlation_id
        ]

        assert len(pipeline_messages) == 3

        # Verify message flow order
        expected_flow = [
            (AgentRole.ARCHITECT, AgentRole.CODER),
            (AgentRole.CODER, AgentRole.TESTER),
            (AgentRole.TESTER, AgentRole.ANALYZER)
        ]

        for i, (expected_from, expected_to) in enumerate(expected_flow):
            msg = pipeline_messages[i]
            assert msg.from_agent == expected_from
            assert msg.to_agent == expected_to

    @pytest.mark.asyncio
    async def test_publish_subscribe_pattern(self, orchestrator):
        """Test publish-subscribe communication pattern"""
        await orchestrator.initialize()

        # Analyzer publishes performance metrics
        performance_update = AgentMessage(
            message_id="perf_update_001",
            from_agent=AgentRole.ANALYZER,
            to_agent=AgentRole.CODER,  # In real system, this would be to multiple subscribers
            message_type="performance_metrics_update",
            content={
                "cpu_usage": 75.5,
                "memory_usage": 62.3,
                "response_time": 125.8,
                "timestamp": datetime.now().isoformat()
            },
            correlation_id="performance_monitoring"
        )

        # Multiple agents could subscribe to these updates
        subscriber_agents = [AgentRole.CODER, AgentRole.DEBUGGER, AgentRole.ARCHITECT]

        # Simulate sending to multiple subscribers
        for i, subscriber in enumerate(subscriber_agents):
            subscriber_message = AgentMessage(
                message_id=f"perf_update_00{i+1}",
                from_agent=AgentRole.ANALYZER,
                to_agent=subscriber,
                message_type="performance_metrics_update",
                content=performance_update.content,
                correlation_id="performance_monitoring"
            )
            await orchestrator.send_message(subscriber_message)

        # Verify all subscribers received the update
        perf_messages = [
            msg for msg in orchestrator.message_history
            if msg.correlation_id == "performance_monitoring"
        ]

        assert len(perf_messages) == 3
        assert all(msg.from_agent == AgentRole.ANALYZER for msg in perf_messages)
        assert all(msg.message_type == "performance_metrics_update" for msg in perf_messages)


class TestCoordinationScenarios:
    """Tests for complex coordination scenarios"""

    @pytest.fixture
    def integration_test_data(self):
        """Load integration test data"""
        return TEST_DATA.get('test_scenarios', {}).get('integration_scenarios', [])

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_scenario(self, orchestrator, integration_test_data):
        """Test multi-agent coordination scenario from test data"""
        if not integration_test_data:
            pytest.skip("Integration test data not available")

        await orchestrator.initialize()

        # Find multi-agent coordination scenario
        coordination_scenario = next(
            (scenario for scenario in integration_test_data
             if scenario.get('name') == 'Multi-Agent Coordination'),
            None
        )

        if not coordination_scenario:
            pytest.skip("Multi-agent coordination scenario not found in test data")

        # Execute the scenario steps
        agents_involved = coordination_scenario.get('agents_involved', [])
        scenario_steps = coordination_scenario.get('scenario_steps', [])

        # Create tasks for each step
        tasks = []
        for i, step in enumerate(scenario_steps):
            # Map step to appropriate agent
            agent_mapping = {
                "coder": AgentRole.CODER,
                "tester": AgentRole.TESTER,
                "debugger": AgentRole.DEBUGGER
            }

            # Get agent for this step (simplified mapping)
            agent_role = None
            for agent_name, role in agent_mapping.items():
                if agent_name in step.lower():
                    agent_role = role
                    break

            if not agent_role:
                agent_role = AgentRole.CODER  # Default

            task = AgentTask(
                task_id=f"scenario_task_{i}",
                task_type="execute_step",
                description=step,
                assigned_agent=agent_role,
                priority=5,
                parameters={"step_description": step}
            )

            tasks.append(task)

        # Execute tasks
        results = []
        for task in tasks:
            await orchestrator.assign_task(task)
            result = await orchestrator.execute_task(task.task_id)
            results.append(result)

        # Verify all tasks completed
        assert len(results) == len(tasks)
        assert all(result["status"] == TaskStatus.COMPLETED.value for result in results)

        # Check expected interactions
        expected_interactions = coordination_scenario.get('expected_interactions', 0)
        # In a real test, we'd verify the actual interaction count matches expected
        assert len(results) >= expected_interactions

    @pytest.mark.asyncio
    async def test_complex_feature_development_workflow(self, orchestrator):
        """Test complex feature development with all agents"""
        await orchestrator.initialize()

        # Define a complex feature development workflow
        feature_name = "advanced_search_system"

        # Create comprehensive task set
        tasks = [
            # Architecture design
            AgentTask(
                task_id="arch_design",
                task_type="design_system_architecture",
                description=f"Design architecture for {feature_name}",
                assigned_agent=AgentRole.ARCHITECT,
                priority=9,
                parameters={"feature": feature_name, "complexity": "high"}
            ),

            # Core implementation
            AgentTask(
                task_id="core_impl",
                task_type="implement_core_logic",
                description=f"Implement core logic for {feature_name}",
                assigned_agent=AgentRole.CODER,
                priority=8,
                dependencies=["arch_design"],
                parameters={"architecture": "from_design", "language": "python"}
            ),

            # UI implementation
            AgentTask(
                task_id="ui_impl",
                task_type="implement_user_interface",
                description=f"Implement UI for {feature_name}",
                assigned_agent=AgentRole.UI_REFINER,
                priority=7,
                dependencies=["arch_design"],
                parameters={"interface_type": "web", "accessibility": "WCAG_2.1"}
            ),

            # Performance analysis
            AgentTask(
                task_id="perf_analysis",
                task_type="analyze_performance",
                description=f"Analyze performance of {feature_name}",
                assigned_agent=AgentRole.ANALYZER,
                priority=6,
                dependencies=["core_impl"],
                parameters={"analysis_type": "comprehensive", "target_response_time": "100ms"}
            ),

            # Comprehensive testing
            AgentTask(
                task_id="comprehensive_test",
                task_type="create_comprehensive_tests",
                description=f"Create comprehensive test suite for {feature_name}",
                assigned_agent=AgentRole.TESTER,
                priority=7,
                dependencies=["core_impl", "ui_impl"],
                parameters={"test_types": ["unit", "integration", "e2e"], "coverage_target": 95}
            ),

            # Bug fixing (if needed)
            AgentTask(
                task_id="bug_analysis",
                task_type="analyze_potential_issues",
                description=f"Analyze potential issues in {feature_name}",
                assigned_agent=AgentRole.DEBUGGER,
                priority=6,
                dependencies=["comprehensive_test"],
                parameters={"analysis_scope": "full_system"}
            )
        ]

        # Assign all tasks
        for task in tasks:
            await orchestrator.assign_task(task)

        # Execute tasks respecting dependencies
        completed_tasks = set()
        remaining_tasks = {task.task_id: task for task in tasks}

        while remaining_tasks:
            # Find tasks that can be executed (dependencies satisfied)
            executable_tasks = []
            for task_id, task in remaining_tasks.items():
                if all(dep in completed_tasks for dep in task.dependencies):
                    executable_tasks.append(task_id)

            if not executable_tasks:
                break  # No more tasks can be executed (should not happen in well-formed workflow)

            # Execute available tasks
            for task_id in executable_tasks:
                result = await orchestrator.execute_task(task_id)
                assert result["status"] == TaskStatus.COMPLETED.value
                completed_tasks.add(task_id)
                del remaining_tasks[task_id]

        # Verify all tasks completed
        assert len(completed_tasks) == len(tasks)
        assert len(remaining_tasks) == 0

        # Verify coordination metrics
        metrics = await orchestrator.get_coordination_metrics()
        assert metrics["tasks_completed"] == len(tasks)
        assert metrics["task_completion_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_emergency_coordination_scenario(self, orchestrator):
        """Test emergency coordination scenario (e.g., critical bug fixing)"""
        await orchestrator.initialize()

        # Simulate critical bug scenario requiring immediate coordination
        emergency_parameters = {
            "bug_severity": "critical",
            "affected_systems": ["user_authentication", "payment_processing"],
            "user_impact": "high",
            "time_constraint": "2_hours"
        }

        # Emergency coordination should involve multiple agents quickly
        coordination_start = datetime.now()
        result = await orchestrator.coordinate_agents("bug_fixing_workflow", emergency_parameters)
        coordination_duration = (datetime.now() - coordination_start).total_seconds()

        assert result["outcome"] == "success"
        assert coordination_duration < 10.0  # Emergency scenarios should coordinate quickly

        # Verify rapid response coordination
        assert len(result["participants"]) >= 3  # Multiple agents involved
        assert AgentRole.DEBUGGER.value in result["participants"]  # Debugger leads emergency response

        # Should have immediate communication
        assert len(result["messages_exchanged"]) > 0

    @pytest.mark.asyncio
    async def test_resource_contention_handling(self, orchestrator):
        """Test handling of resource contention between agents"""
        await orchestrator.initialize()

        # Create multiple tasks that might compete for the same conceptual resources
        competing_tasks = [
            AgentTask(
                task_id="resource_task_1",
                task_type="modify_core_module",
                description="Modify core authentication module",
                assigned_agent=AgentRole.CODER,
                priority=8,
                parameters={"module": "auth_core", "operation": "refactor"}
            ),
            AgentTask(
                task_id="resource_task_2",
                task_type="test_core_module",
                description="Test core authentication module",
                assigned_agent=AgentRole.TESTER,
                priority=8,
                parameters={"module": "auth_core", "test_type": "comprehensive"}
            ),
            AgentTask(
                task_id="resource_task_3",
                task_type="analyze_core_module",
                description="Analyze core authentication module performance",
                assigned_agent=AgentRole.ANALYZER,
                priority=8,
                parameters={"module": "auth_core", "analysis_depth": "deep"}
            )
        ]

        # Assign and execute competing tasks
        task_ids = []
        for task in competing_tasks:
            task_id = await orchestrator.assign_task(task)
            task_ids.append(task_id)

        # Execute tasks (in real system, resource coordination would be handled)
        execution_tasks = [orchestrator.execute_task(tid) for tid in task_ids]
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)

        # All tasks should complete (mock system doesn't have real resource contention)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 2  # At least most tasks should succeed

        # In a real system, this would test resource locking, queuing, etc.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])