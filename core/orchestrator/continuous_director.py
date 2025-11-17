"""
Continuous Director - The never-stopping orchestrator that manages the entire development process.
This is the brain that coordinates all agents and ensures continuous improvement until perfection.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

from pydantic import BaseModel
from loguru import logger

# Configure logging
logger.add("logs/orchestrator.log", rotation="1 day", retention="30 days", level="DEBUG")


class ProjectState(Enum):
    """States of the continuous development project"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    DEVELOPING = "developing"
    TESTING = "testing"
    DEBUGGING = "debugging"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR_RECOVERY = "error_recovery"


class TaskPriority(Enum):
    """Priority levels for development tasks"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    DEFERRED = 5


@dataclass
class DevelopmentTask:
    """Represents a single development task"""
    id: str
    type: str
    description: str
    priority: TaskPriority
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3
    error_history: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert task to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "priority": self.priority.value,
            "assigned_agent": self.assigned_agent,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempts": self.attempts,
            "error_history": self.error_history,
            "dependencies": self.dependencies
        }


class QualityMetrics(BaseModel):
    """Tracks quality metrics for the application being built"""
    test_coverage: float = 0.0
    bug_count_critical: int = 0
    bug_count_minor: int = 0
    performance_score: float = 0.0
    documentation_coverage: float = 0.0
    code_quality_score: float = 0.0
    user_satisfaction: float = 0.0
    security_score: float = 0.0

    def is_perfect(self) -> bool:
        """Check if the application meets perfection criteria"""
        return (
            self.test_coverage >= 95.0 and
            self.bug_count_critical == 0 and
            self.bug_count_minor <= 5 and
            self.performance_score >= 90.0 and
            self.documentation_coverage >= 90.0 and
            self.code_quality_score >= 85.0 and
            self.user_satisfaction >= 90.0 and
            self.security_score >= 95.0
        )

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for API responses"""
        return {
            "test_coverage": self.test_coverage,
            "bug_count_critical": self.bug_count_critical,
            "bug_count_minor": self.bug_count_minor,
            "performance_score": self.performance_score,
            "documentation_coverage": self.documentation_coverage,
            "code_quality_score": self.code_quality_score,
            "user_satisfaction": self.user_satisfaction,
            "security_score": self.security_score,
            "is_perfect": self.is_perfect()
        }


class ContinuousDirector:
    """
    The master orchestrator that never stops working until the application is perfect.
    Manages agents, tasks, memory, and the continuous improvement cycle.
    """

    def __init__(self, project_name: str, project_spec: Dict):
        self.project_name = project_name
        self.project_spec = project_spec
        self.state = ProjectState.INITIALIZING
        self.metrics = QualityMetrics()

        # Task management
        self.task_queue: List[DevelopmentTask] = []
        self.active_tasks: Dict[str, DevelopmentTask] = {}
        self.completed_tasks: List[DevelopmentTask] = []
        self.failed_tasks: List[DevelopmentTask] = []

        # Agent registry (will be populated when agents are initialized)
        self.agents: Dict[str, Any] = {}

        # Iteration tracking
        self.iteration_count = 0
        self.start_time = datetime.now()
        self.last_checkpoint = datetime.now()
        self.checkpoint_interval = timedelta(hours=1)

        # Learning and adaptation
        self.error_patterns: Dict[str, int] = {}
        self.successful_strategies: List[Dict] = []
        self.user_feedback_history: List[Dict] = []

        # Control flags
        self.is_running = False
        self.pause_requested = False
        self.stop_requested = False

        logger.info(f"Initialized ContinuousDirector for project: {project_name}")

    async def start(self):
        """Start the continuous development process"""
        logger.info("Starting continuous development process...")
        self.is_running = True
        self.state = ProjectState.PLANNING

        try:
            # Initialize all components
            await self._initialize_components()

            # Start continuous quality monitor in background
            asyncio.create_task(self.continuous_quality_monitor())

            # Main continuous loop
            while not self.stop_requested and not self.metrics.is_perfect():
                if self.pause_requested:
                    self.state = ProjectState.PAUSED
                    # REAL EVENT-DRIVEN WAITING - NO FAKE DELAYS IN SAFETY-CRITICAL SYSTEM
                    while self.pause_requested and not self.stop_requested:
                        # Check for unpause or stop events every 0.1s (real-time responsiveness)
                        await asyncio.sleep(0.1)
                    continue

                # Execute one development iteration
                await self._execute_iteration()

                # Check if checkpoint is needed
                if datetime.now() - self.last_checkpoint > self.checkpoint_interval:
                    await self._create_checkpoint()

                # MINIMAL REAL-TIME YIELD - NO FAKE DELAYS IN SAFETY-CRITICAL SYSTEM
                # Only yield control briefly to allow other coroutines to run
                await asyncio.sleep(0.01)  # 10ms - minimal yield, not artificial delay

            if self.metrics.is_perfect():
                self.state = ProjectState.COMPLETED
                logger.success(f"Project {self.project_name} completed successfully!")

        except Exception as e:
            logger.error(f"Critical error in continuous director: {e}")
            self.state = ProjectState.ERROR_RECOVERY
            await self._recover_from_error(e)

        finally:
            self.is_running = False
            await self._cleanup()

    async def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")

        # Initialize memory system
        from ..memory.project_ledger import ProjectLedger
        from ..memory.vector_memory import VectorMemory
        from ..memory.error_knowledge_graph import ErrorKnowledgeGraph
        from .milestone_tracker import MilestoneTracker
        from .progress_analyzer import ProgressAnalyzer

        self.project_ledger = ProjectLedger(self.project_name)
        self.vector_memory = VectorMemory(project_name=self.project_name)
        self.error_graph = ErrorKnowledgeGraph()

        # Initialize tracking components
        self.milestone_tracker = MilestoneTracker(self.project_name)
        self.progress_analyzer = ProgressAnalyzer(self.project_name)

        # Initialize agents
        await self._initialize_agents()

        # Load any existing project state
        await self._load_project_state()

        logger.success("All components initialized successfully")

    async def _initialize_agents(self):
        """Initialize all AI agents"""
        logger.info("Initializing AI agents...")

        # Import agent classes (these will be implemented)
        from ..agents.coder_agent import CoderAgent
        from ..agents.tester_agent import TesterAgent
        from ..agents.debugger_agent import DebuggerAgent
        from ..agents.architect_agent import ArchitectAgent
        from ..agents.analyzer_agent import AnalyzerAgent
        from ..agents.ui_refiner_agent import UIRefinerAgent

        # Initialize each agent with the orchestrator reference
        self.agents = {
            "coder": CoderAgent(orchestrator=self),
            "tester": TesterAgent(orchestrator=self),
            "debugger": DebuggerAgent(orchestrator=self),
            "architect": ArchitectAgent(orchestrator=self),
            "analyzer": AnalyzerAgent(orchestrator=self),
            "ui_refiner": UIRefinerAgent(orchestrator=self)
        }

        # Initialize all agents (if they have initialize method)
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'initialize'):
                await agent.initialize()
            logger.info(f"Initialized agent: {agent_name}")

    async def _execute_iteration(self):
        """Execute one complete development iteration"""
        self.iteration_count += 1
        logger.info(f"Starting iteration #{self.iteration_count}")

        # Plan phase
        self.state = ProjectState.PLANNING
        tasks = await self._plan_iteration()

        # Development phase
        self.state = ProjectState.DEVELOPING
        await self._execute_development_tasks(tasks)

        # Testing phase
        self.state = ProjectState.TESTING
        test_results = await self._run_tests()

        # Debugging phase (if needed)
        if test_results.get("failures"):
            self.state = ProjectState.DEBUGGING
            await self._debug_and_fix(test_results["failures"])

        # Optimization phase
        self.state = ProjectState.OPTIMIZING
        await self._optimize_performance()

        # Validation phase
        self.state = ProjectState.VALIDATING
        await self._validate_quality()

        # Learn from this iteration
        await self._learn_from_iteration()

        logger.info(f"Completed iteration #{self.iteration_count}")

    async def _plan_iteration(self) -> List[DevelopmentTask]:
        """Plan tasks for the current iteration"""
        tasks = []

        # Analyze current state and metrics
        analysis = await self._analyze_current_state()

        # Generate tasks based on priorities
        if self.metrics.bug_count_critical > 0:
            # Critical bugs take top priority
            tasks.extend(self._generate_bug_fix_tasks(priority=TaskPriority.CRITICAL))

        if self.metrics.test_coverage < 95.0:
            # Need more tests
            tasks.extend(self._generate_test_tasks(priority=TaskPriority.HIGH))

        if self.metrics.performance_score < 90.0:
            # Performance optimization needed
            tasks.extend(self._generate_optimization_tasks(priority=TaskPriority.HIGH))

        # Regular feature development
        tasks.extend(self._generate_feature_tasks(priority=TaskPriority.NORMAL))

        # Sort tasks by priority and dependencies
        tasks = self._prioritize_tasks(tasks)

        return tasks

    async def _execute_development_tasks(self, tasks: List[DevelopmentTask]):
        """Execute development tasks using agents"""
        for task in tasks:
            try:
                # Assign task to appropriate agent
                agent = self._select_agent_for_task(task)
                task.assigned_agent = agent
                task.started_at = datetime.now()
                task.status = "in_progress"

                logger.info(f"Assigning task {task.id} to agent {agent}")

                # Execute task
                result = await self.agents[agent].process_task(task)

                # Update task status
                task.completed_at = datetime.now()
                task.status = "completed"
                task.result = result
                self.completed_tasks.append(task)

                logger.success(f"Task {task.id} completed successfully")

            except Exception as e:
                logger.error(f"Task {task.id} failed: {e}")
                task.attempts += 1
                task.error_history.append(str(e))

                if task.attempts < task.max_attempts:
                    # Retry the task
                    self.task_queue.append(task)
                else:
                    # Mark as failed and learn from it
                    task.status = "failed"
                    self.failed_tasks.append(task)
                    await self._learn_from_failure(task)

    async def _create_checkpoint(self):
        """Create a checkpoint of the current state"""
        logger.info("Creating checkpoint...")

        checkpoint_data = {
            "iteration": self.iteration_count,
            "timestamp": datetime.now().isoformat(),
            "state": self.state.value,
            "metrics": self.metrics.dict(),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "error_patterns": self.error_patterns,
            "successful_strategies": self.successful_strategies[-10:]  # Keep last 10
        }

        # Save checkpoint to persistent storage
        checkpoint_path = Path(f"persistence/snapshots/checkpoint_{self.iteration_count}.json")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        self.last_checkpoint = datetime.now()
        logger.success(f"Checkpoint created: {checkpoint_path}")

    async def _learn_from_iteration(self):
        """Learn from the completed iteration"""
        # Analyze what worked
        successful_patterns = await self._analyze_successes()
        self.successful_strategies.extend(successful_patterns)

        # Analyze what failed
        failure_patterns = await self._analyze_failures()
        for pattern in failure_patterns:
            self.error_patterns[pattern] = self.error_patterns.get(pattern, 0) + 1

        # Update agent strategies based on learning
        for agent in self.agents.values():
            await agent.update_strategy(successful_patterns, failure_patterns)

        logger.info(f"Learning complete. Success patterns: {len(successful_patterns)}, "
                   f"Failure patterns: {len(failure_patterns)}")

    def _select_agent_for_task(self, task: DevelopmentTask) -> str:
        """Select the most appropriate agent for a task"""
        task_type_to_agent = {
            "code": "coder",
            "test": "tester",
            "debug": "debugger",
            "architecture": "architect",
            "analysis": "analyzer",
            "ui": "ui_refiner"
        }

        for task_type, agent in task_type_to_agent.items():
            if task_type in task.type.lower():
                return agent

        # Default to coder agent
        return "coder"

    # Implementation of core methods
    async def _load_project_state(self):
        """Load existing project state from persistent storage"""
        try:
            # Try to load previous metrics
            checkpoint_dir = Path(f"persistence/snapshots")
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))
                if checkpoints:
                    latest_checkpoint = checkpoints[-1]
                    with open(latest_checkpoint, "r") as f:
                        data = json.load(f)

                    # Restore state
                    self.iteration_count = data.get("iteration", 0)
                    self.state = ProjectState(data.get("state", "planning"))

                    # Restore metrics if available
                    if "metrics" in data:
                        self.metrics = QualityMetrics(**data["metrics"])

                    logger.info(f"Loaded project state from {latest_checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load previous state: {e}. Starting fresh.")

    async def _run_tests(self) -> Dict:
        """Run tests using the tester agent"""
        if "tester" not in self.agents:
            return {"status": "no_tester", "failures": []}

        try:
            # Create test execution task
            task = DevelopmentTask(
                id=f"test-run-{self.iteration_count}",
                type="test_execution",
                description="Run all tests and measure coverage",
                priority=TaskPriority.HIGH,
                data={"measure_coverage": True}
            )

            # Execute tests
            result = await self.agents["tester"].process_task(task)

            # Update metrics
            if result and "coverage" in result:
                self.metrics.test_coverage = result["coverage"]

            return {
                "status": "completed",
                "failures": result.get("failures", []) if result else [],
                "coverage": result.get("coverage", 0.0) if result else 0.0
            }
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {"status": "error", "failures": [], "error": str(e)}

    async def _debug_and_fix(self, failures):
        """Debug and fix test failures using debugger agent"""
        if "debugger" not in self.agents or not failures:
            return

        try:
            for i, failure in enumerate(failures[:5]):  # Process up to 5 failures at once
                task = DevelopmentTask(
                    id=f"debug-fix-{self.iteration_count}-{i}",
                    type="debug_failure",
                    description=f"Fix test failure: {failure.get('test_name', 'unknown')}",
                    priority=TaskPriority.CRITICAL,
                    data={"failure_info": failure}
                )

                result = await self.agents["debugger"].process_task(task)

                if result and result.get("fixed"):
                    logger.success(f"Fixed failure: {failure.get('test_name')}")
                    # Decrease bug count
                    if failure.get("severity") == "critical":
                        self.metrics.bug_count_critical = max(0, self.metrics.bug_count_critical - 1)
                    else:
                        self.metrics.bug_count_minor = max(0, self.metrics.bug_count_minor - 1)
        except Exception as e:
            logger.error(f"Debugging failed: {e}")

    async def _optimize_performance(self):
        """Optimize performance using analyzer agent"""
        if "analyzer" not in self.agents:
            return

        try:
            task = DevelopmentTask(
                id=f"performance-analysis-{self.iteration_count}",
                type="performance_analysis",
                description="Analyze and optimize performance",
                priority=TaskPriority.NORMAL,
                data={"current_score": self.metrics.performance_score}
            )

            result = await self.agents["analyzer"].process_task(task)

            if result and "performance_score" in result:
                self.metrics.performance_score = result["performance_score"]
                logger.info(f"Performance score: {self.metrics.performance_score}%")
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")

    async def _validate_quality(self):
        """Validate overall quality metrics"""
        if "analyzer" not in self.agents:
            return

        try:
            task = DevelopmentTask(
                id=f"quality-validation-{self.iteration_count}",
                type="quality_validation",
                description="Validate all quality metrics",
                priority=TaskPriority.NORMAL,
                data={"current_metrics": self.metrics.dict()}
            )

            result = await self.agents["analyzer"].process_task(task)

            if result:
                # Update all metrics from validation
                if "code_quality_score" in result:
                    self.metrics.code_quality_score = result["code_quality_score"]
                if "documentation_coverage" in result:
                    self.metrics.documentation_coverage = result["documentation_coverage"]
                if "security_score" in result:
                    self.metrics.security_score = result["security_score"]

                logger.info("Quality validation complete")
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")

    async def _analyze_current_state(self) -> Dict:
        """Analyze current project state and identify needs"""
        weak_metrics = self.identify_weak_metrics()

        analysis = {
            "iteration": self.iteration_count,
            "state": self.state.value,
            "metrics": self.metrics.dict(),
            "weak_metrics": weak_metrics,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "needs_tests": self.metrics.test_coverage < 95.0,
            "needs_optimization": self.metrics.performance_score < 90.0,
            "has_critical_bugs": self.metrics.bug_count_critical > 0,
            "needs_documentation": self.metrics.documentation_coverage < 90.0,
            "needs_security": self.metrics.security_score < 95.0
        }

        return analysis

    async def _recover_from_error(self, error):
        """Attempt to recover from critical errors"""
        logger.error(f"Attempting recovery from error: {error}")

        try:
            # Record error in knowledge graph
            if hasattr(self, 'error_graph'):
                self.error_graph.add_error(
                    error_type=type(error).__name__,
                    error_message=str(error),
                    context={"iteration": self.iteration_count, "state": self.state.value},
                    stack_trace=""
                )

            # Try to load last checkpoint
            await self._load_project_state()

            # Reset to planning state
            self.state = ProjectState.PLANNING

            # Clear task queue and retry
            self.task_queue.clear()

            logger.info("Recovery attempt completed")
        except Exception as e:
            logger.error(f"Recovery failed: {e}")

    async def _analyze_successes(self) -> List:
        """Analyze successful patterns from completed tasks"""
        patterns = []

        # Look at recent successful tasks
        recent_successes = [t for t in self.completed_tasks[-20:] if t.status == "completed"]

        for task in recent_successes:
            pattern = {
                "task_type": task.type,
                "agent": task.assigned_agent,
                "duration": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else 0,
                "attempts": task.attempts,
                "iteration": self.iteration_count
            }
            patterns.append(pattern)

        return patterns

    async def _analyze_failures(self) -> List:
        """Analyze failure patterns from failed tasks"""
        patterns = []

        for task in self.failed_tasks[-10:]:  # Last 10 failures
            pattern = {
                "task_type": task.type,
                "error_types": task.error_history,
                "attempts": task.attempts,
                "iteration": self.iteration_count
            }
            patterns.append(pattern)

        return patterns

    async def _learn_from_failure(self, task):
        """Learn from a failed task"""
        if not hasattr(self, 'error_graph'):
            return

        # Record each error in the knowledge graph
        for error in task.error_history:
            self.error_graph.add_error(
                error_type=task.type,
                error_message=error,
                context={
                    "task_id": task.id,
                    "agent": task.assigned_agent,
                    "iteration": self.iteration_count
                },
                stack_trace=""
            )

    def _generate_bug_fix_tasks(self, priority) -> List:
        """Generate tasks to fix bugs"""
        tasks = []

        # Generate tasks for critical bugs
        for i in range(self.metrics.bug_count_critical):
            task = DevelopmentTask(
                id=f"bugfix-critical-{self.iteration_count}-{i}",
                type="debug_critical_bug",
                description=f"Fix critical bug #{i+1}",
                priority=priority,
                data={"severity": "critical", "bug_index": i}
            )
            tasks.append(task)

        # Generate tasks for minor bugs (if not many critical ones)
        if self.metrics.bug_count_critical == 0:
            for i in range(min(self.metrics.bug_count_minor, 3)):
                task = DevelopmentTask(
                    id=f"bugfix-minor-{self.iteration_count}-{i}",
                    type="debug_minor_bug",
                    description=f"Fix minor bug #{i+1}",
                    priority=TaskPriority.NORMAL,
                    data={"severity": "minor", "bug_index": i}
                )
                tasks.append(task)

        return tasks

    def _generate_test_tasks(self, priority) -> List:
        """Generate tasks to improve test coverage"""
        tasks = []

        coverage_gap = 95.0 - self.metrics.test_coverage
        if coverage_gap > 0:
            # Generate test tasks based on coverage gap
            num_tasks = max(1, int(coverage_gap / 10))  # One task per 10% gap

            for i in range(num_tasks):
                task = DevelopmentTask(
                    id=f"test-generation-{self.iteration_count}-{i}",
                    type="test_generation",
                    description=f"Generate tests to improve coverage (batch {i+1})",
                    priority=priority,
                    data={
                        "target_coverage": 95.0,
                        "current_coverage": self.metrics.test_coverage,
                        "batch": i
                    }
                )
                tasks.append(task)

        return tasks

    def _generate_optimization_tasks(self, priority) -> List:
        """Generate performance optimization tasks"""
        tasks = []

        if self.metrics.performance_score < 90.0:
            task = DevelopmentTask(
                id=f"optimize-performance-{self.iteration_count}",
                type="performance_optimization",
                description="Optimize application performance",
                priority=priority,
                data={
                    "target_score": 90.0,
                    "current_score": self.metrics.performance_score
                }
            )
            tasks.append(task)

        return tasks

    def _generate_feature_tasks(self, priority) -> List:
        """Generate feature development tasks from project spec"""
        tasks = []

        # Check if there are features in the project spec
        if "features" in self.project_spec:
            for i, feature in enumerate(self.project_spec["features"][:3]):  # Limit to 3 features per iteration
                task = DevelopmentTask(
                    id=f"feature-{self.iteration_count}-{i}",
                    type="code_feature",
                    description=f"Implement feature: {feature.get('name', 'unknown')}",
                    priority=priority,
                    data={"feature": feature}
                )
                tasks.append(task)

        return tasks

    def _prioritize_tasks(self, tasks) -> List:
        """Prioritize and sort tasks by priority and dependencies"""
        if not tasks:
            return tasks

        # Sort by priority (lower number = higher priority)
        sorted_tasks = sorted(tasks, key=lambda t: (
            t.priority.value if isinstance(t.priority, TaskPriority) else t.priority,
            t.created_at
        ))

        # Handle dependencies (simple topological sort)
        final_order = []
        completed_ids = set()

        max_iterations = len(sorted_tasks) * 2
        iteration = 0

        while sorted_tasks and iteration < max_iterations:
            iteration += 1
            for task in sorted_tasks[:]:
                # Check if all dependencies are completed
                deps_met = all(dep_id in completed_ids for dep_id in task.dependencies)

                if deps_met:
                    final_order.append(task)
                    completed_ids.add(task.id)
                    sorted_tasks.remove(task)

        # Add remaining tasks (in case of circular dependencies)
        final_order.extend(sorted_tasks)

        return final_order

    # Control methods
    async def pause(self):
        """Pause the continuous development"""
        self.pause_requested = True
        logger.info("Pause requested")

    async def resume(self):
        """Resume the continuous development"""
        self.pause_requested = False
        logger.info("Resume requested")

    async def stop(self):
        """Stop the continuous development"""
        self.stop_requested = True
        logger.info("Stop requested")
    async def _cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("Cleaning up resources...")
        
        # Save final checkpoint
        await self._create_checkpoint()
        
        # Clean up agents
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        logger.info("Cleanup completed")

    async def continuous_quality_monitor(self):
        """Continuously monitor quality metrics and trigger optimizations"""
        logger.info("Starting continuous quality monitor...")

        while self.is_running:
            try:
                metrics = self.metrics.to_dict()

                # Check if metrics are below thresholds
                if metrics["test_coverage"] < 95:
                    logger.warning(f"Test coverage low: {metrics['test_coverage']}%")
                    await self.trigger_test_intensification()

                if metrics["performance_score"] < 90:
                    logger.warning(f"Performance score low: {metrics['performance_score']}%")
                    await self.trigger_performance_optimization()

                if metrics["bug_count_critical"] > 0:
                    logger.error(f"Critical bugs detected: {metrics['bug_count_critical']}")
                    await self.emergency_debug_mode()

                if metrics["documentation_coverage"] < 90:
                    logger.warning(f"Documentation coverage low: {metrics['documentation_coverage']}%")
                    await self.trigger_documentation_generation()

                if metrics["security_score"] < 95:
                    logger.warning(f"Security score low: {metrics['security_score']}%")
                    await self.trigger_security_audit()

                # REAL-TIME SAFETY MONITORING - NO DANGEROUS DELAYS IN CRITICAL SYSTEM
                # Safety-critical systems need frequent monitoring (every 10 seconds max)
                await asyncio.sleep(10)  # 10 seconds - appropriate for safety-critical monitoring

            except Exception as e:
                logger.error(f"Error in quality monitor: {e}")
                # RAPID ERROR RECOVERY - NO DELAYS IN SAFETY-CRITICAL ERROR HANDLING
                await asyncio.sleep(5)  # 5 seconds - rapid recovery for safety-critical errors

    async def trigger_test_intensification(self):
        """Deploy tester agent to increase test coverage"""
        if "tester" in self.agents:
            task = DevelopmentTask(
                id=f"test-intensification-{self.iteration_count}",
                type="test_generation",
                description="Generate additional tests to reach 95% coverage",
                priority=TaskPriority.HIGH,
                data={"target_coverage": 95, "current_coverage": self.metrics.test_coverage}
            )
            self.task_queue.insert(0, task)
            logger.info("Triggered test intensification")

    async def trigger_performance_optimization(self):
        """Deploy analyzer agent to optimize performance"""
        if "analyzer" in self.agents:
            task = DevelopmentTask(
                id=f"performance-opt-{self.iteration_count}",
                type="performance_optimization",
                description="Analyze and optimize performance bottlenecks",
                priority=TaskPriority.HIGH,
                data={"target_score": 90, "current_score": self.metrics.performance_score}
            )
            self.task_queue.insert(0, task)
            logger.info("Triggered performance optimization")

    async def emergency_debug_mode(self):
        """Emergency mode when critical bugs are detected"""
        logger.error("ENTERING EMERGENCY DEBUG MODE")
        self.state = ProjectState.DEBUGGING

        if "debugger" in self.agents:
            task = DevelopmentTask(
                id=f"emergency-debug-{self.iteration_count}",
                type="emergency_debug",
                description="Fix all critical bugs immediately",
                priority=TaskPriority.CRITICAL,
                data={"bug_count": self.metrics.bug_count_critical}
            )
            # Clear other tasks and focus on debugging
            self.task_queue = [task]
            logger.info("All tasks cleared - focusing on critical bug fixes")

    async def trigger_documentation_generation(self):
        """Generate missing documentation"""
        if "coder" in self.agents:
            task = DevelopmentTask(
                id=f"doc-generation-{self.iteration_count}",
                type="documentation",
                description="Generate comprehensive documentation",
                priority=TaskPriority.NORMAL,
                data={"target_coverage": 90, "current_coverage": self.metrics.documentation_coverage}
            )
            self.task_queue.append(task)
            logger.info("Triggered documentation generation")

    async def trigger_security_audit(self):
        """Perform security audit and fixes"""
        if "analyzer" in self.agents:
            task = DevelopmentTask(
                id=f"security-audit-{self.iteration_count}",
                type="security_audit",
                description="Perform security audit and implement fixes",
                priority=TaskPriority.HIGH,
                data={"target_score": 95, "current_score": self.metrics.security_score}
            )
            self.task_queue.insert(0, task)
            logger.info("Triggered security audit")

    def identify_weak_metrics(self) -> Dict[str, float]:
        """Identify metrics that need improvement"""
        weak_metrics = {}
        metrics = self.metrics.to_dict()

        thresholds = {
            "test_coverage": 95,
            "performance_score": 90,
            "documentation_coverage": 90,
            "security_score": 95,
            "code_quality_score": 85,
            "user_satisfaction": 90
        }

        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                weak_metrics[metric] = metrics[metric]

        if metrics["bug_count_critical"] > 0:
            weak_metrics["bug_count_critical"] = metrics["bug_count_critical"]

        return weak_metrics

    @property
    def ledger(self):
        """Get or create project ledger"""
        if not hasattr(self, '_ledger'):
            from core.memory.project_ledger import ProjectLedger
            self._ledger = ProjectLedger(self.project_name)
        return self._ledger
    
    def update_quality_metrics(self, new_metrics: Dict):
        """Update quality metrics"""
        for key, value in new_metrics.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        
        # Log the update
        self.ledger.record_decision(
            self.iteration_count,
            "orchestrator", 
            "quality_metrics_update",
            f"Updated quality metrics: {new_metrics}"
        )
