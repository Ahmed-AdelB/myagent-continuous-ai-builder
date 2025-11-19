"""
Continuous Director - The never-stopping orchestrator that manages the entire development process.
This is the brain that coordinates all agents and ensures continuous improvement until perfection.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

from pydantic import BaseModel
from loguru import logger

# Task validator for preventing false completion claims
from .task_validator import TaskValidator, ValidationResult

# Guardrails for safe autonomous operation
from .guardrails import GuardrailSystem, RiskLevel, GuardrailViolation

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from core.agents.base_agent import AgentTask

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

        # Enhanced stopping criteria (prevents premature exit)
        self.cannot_stop_flag = False  # Set during critical operations
        self.stagnant_iterations = 0  # Iterations without progress
        self.last_progress_metrics = None  # For detecting progress
        self.max_stagnant_iterations = 10  # Stop if no progress for this many
        self.progress_history: List[Dict] = []  # Track progress velocity

        # Task validator (prevents false completion claims)
        self.task_validator = TaskValidator(project_root=Path.cwd())

        # Guardrail system (prevents dangerous operations in autonomous mode)
        self.guardrails = GuardrailSystem(
            autonomous_mode=True,  # Enable autonomous mode restrictions
            project_root=Path.cwd()
        )

        logger.info(f"Initialized ContinuousDirector for project: {project_name}")
        logger.info("Guardrails enabled - dangerous operations will be blocked")

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

            # Main continuous loop with enhanced stopping criteria
            while True:
                # Check if we can stop safely
                can_stop, stop_reason = self._can_stop_safely()
                if can_stop:
                    logger.info(f"Stopping execution: {stop_reason}")
                    break

                if self.pause_requested:
                    self.state = ProjectState.PAUSED
                    # REAL EVENT-DRIVEN WAITING - NO FAKE DELAYS IN SAFETY-CRITICAL SYSTEM
                    while self.pause_requested and not self.stop_requested:
                        # Check for unpause or stop events every 0.1s (real-time responsiveness)
                        await asyncio.sleep(0.1)
                    continue

                # Execute one development iteration
                await self._execute_iteration()

                # Track progress after each iteration (detect stagnation)
                self._track_progress()

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

        # Initialize all agents
        for agent_name, agent in self.agents.items():
            await agent.initialize()
            logger.info(f"Initialized agent: {agent_name}")

        # Initialize Tri-Agent SDLC Orchestrator for critical consensus-based tasks
        logger.info("Initializing Tri-Agent SDLC system...")
        from .tri_agent_sdlc import TriAgentSDLCOrchestrator

        self.tri_agent_sdlc = TriAgentSDLCOrchestrator(
            project_name=self.project_name,
            working_dir=Path.cwd()
        )
        logger.success("Tri-Agent SDLC system initialized (Claude + Codex + Gemini)")

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

    def _should_use_tri_agent_consensus(self, task: DevelopmentTask) -> bool:
        """
        Determine if a task requires tri-agent consensus approval.

        Critical tasks that need 3-agent review:
        - Security-related changes
        - Architecture modifications
        - Production deployments
        - Frontend component changes
        - Documentation updates (requires accuracy)
        - Breaking API changes
        """
        consensus_keywords = [
            "security", "authentication", "authorization",
            "architecture", "refactor", "breaking",
            "frontend", "component", "ui", "test",
            "documentation", "docstring", "readme",
            "deployment", "production", "release"
        ]

        task_text = f"{task.type} {task.description}".lower()

        return any(keyword in task_text for keyword in consensus_keywords)

    async def _execute_task_with_tri_agent(self, task: DevelopmentTask) -> Dict[str, Any]:
        """
        Execute a task using tri-agent SDLC with consensus voting.

        Returns:
            Dict with success status and results
        """
        logger.info(f"Routing task {task.id} to Tri-Agent SDLC for consensus approval")

        # Convert task to work item for tri-agent system
        work_item_id = self.tri_agent_sdlc.add_work_item(
            title=task.description,
            description=f"Type: {task.type}\nPriority: {task.priority.name}\n\n{task.description}",
            priority=task.priority.value,
            file_paths=[],  # Will be determined during requirements phase
            acceptance_criteria=[]  # Will be inferred by Claude
        )

        logger.info(f"Created tri-agent work item: {work_item_id}")

        # Process the work item through full SDLC with consensus voting
        result = await self.tri_agent_sdlc.process_work_item(work_item_id)

        return result

    async def _execute_development_tasks(self, tasks: List[DevelopmentTask]):
        """Execute development tasks using agents"""
        for task in tasks:
            try:
                task.started_at = datetime.now()
                task.status = "in_progress"

                # Check if task requires tri-agent consensus
                if self._should_use_tri_agent_consensus(task):
                    logger.info(f"Task {task.id} requires tri-agent consensus")
                    task.assigned_agent = "tri-agent-sdlc"

                    # Execute via tri-agent SDLC with consensus voting
                    result = await self._execute_task_with_tri_agent(task)
                else:
                    # Regular single-agent execution
                    agent = self._select_agent_for_task(task)
                    task.assigned_agent = agent

                    logger.info(f"Assigning task {task.id} to agent {agent}")

                    # Execute task (agents implement process_task, not execute_task)
                    result = await self.agents[agent].process_task(task)

                # CRITICAL: Validate completion before marking as done
                validation = await self.task_validator.validate_task_completion(task, result)

                if validation.can_mark_complete:
                    # Task genuinely complete with proof
                    task.completed_at = datetime.now()
                    task.status = "completed"
                    task.result = result
                    self.completed_tasks.append(task)

                    logger.success(
                        f"Task {task.id} completed and validated successfully. "
                        f"Checks passed: {', '.join(validation.checks_passed)}"
                    )
                else:
                    # Task claims completion but validation failed
                    logger.warning(
                        f"Task {task.id} claimed completion but validation FAILED. "
                        f"Reasons: {', '.join(validation.failure_reasons)}"
                    )

                    # Treat as attempt failure, retry if attempts remaining
                    task.attempts += 1
                    task.error_history.append(
                        f"Validation failed: {', '.join(validation.failure_reasons)}"
                    )

                    if task.attempts < task.max_attempts:
                        # Retry with validation feedback
                        logger.info(f"Retrying task {task.id} (attempt {task.attempts}/{task.max_attempts})")
                        self.task_queue.append(task)
                    else:
                        # Max attempts reached, mark as failed
                        task.status = "failed"
                        task.result = result
                        self.failed_tasks.append(task)
                        logger.error(f"Task {task.id} failed validation after {task.attempts} attempts")
                        await self._learn_from_failure(task)

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

    # GEMINI-EDIT - 2025-11-18 - Implemented the learning loop methods.
    async def _learn_from_iteration(self):
        """Learn from the completed iteration by analyzing successes and failures."""
        logger.info("Executing learning phase...")
        
        # Analyze what worked
        successful_patterns = await self._analyze_successes()
        self.successful_strategies.extend(successful_patterns)

        # Analyze what failed and populate the knowledge graph
        failure_patterns = await self._analyze_failures()
        for pattern in failure_patterns:
            self.error_patterns[pattern] = self.error_patterns.get(pattern, 0) + 1

        # A more advanced implementation would have agents update their internal
        # strategies based on these learned patterns.
        # for agent in self.agents.values():
        #     await agent.update_strategy(successful_patterns, failure_patterns)

        logger.info(f"Learning complete. Success patterns: {len(successful_patterns)}, Failure patterns: {len(failure_patterns)}")

    async def _analyze_successes(self) -> List[Dict]:
        """Analyzes successful tasks to identify effective strategies."""
        # This is a simplified implementation. A more advanced version would
        # link successful fixes back to the errors they solved in the graph.
        successful_patterns = []
        for task in self.completed_tasks:
            if task.status == "completed" and not task.error_history:
                pattern = {
                    "task_type": task.type,
                    "assigned_agent": task.assigned_agent,
                    "description": task.description,
                }
                successful_patterns.append(pattern)
        logger.info(f"Identified {len(successful_patterns)} successful patterns in this iteration.")
        return successful_patterns

    async def _analyze_failures(self) -> List[str]:
        """Analyzes failed tasks and populates the ErrorKnowledgeGraph."""
        failure_patterns = []
        for task in self.failed_tasks:
            if task.error_history:
                last_error = task.error_history[-1]
                # A more robust implementation would parse the error more deeply
                error_type = "TaskExecutionError"
                
                # Add the error to the knowledge graph
                self.error_graph.add_error(
                    error_type=error_type,
                    error_message=last_error,
                    context={
                        "task_id": task.id,
                        "task_type": task.type,
                        "assigned_agent": task.assigned_agent,
                    }
                )
                failure_patterns.append(error_type)
        
        # Clear the list of failed tasks for the next iteration
        self.failed_tasks.clear()
        logger.info(f"Analyzed and recorded {len(failure_patterns)} failures in the knowledge graph.")
        return failure_patterns

    async def _learn_from_failure(self, task: DevelopmentTask):
        """Record a single, permanently failed task in the knowledge graph."""
        if not task.error_history:
            return
            
        last_error = task.error_history[-1]
        error_type = "PermanentTaskFailure"
        
        self.error_graph.add_error(
            error_type=error_type,
            error_message=last_error,
            context={
                "task_id": task.id,
                "task_type": task.type,
                "assigned_agent": task.assigned_agent,
                "attempts": task.attempts,
            }
        )
        logger.warning(f"Task {task.id} failed permanently and was recorded in the knowledge graph.")

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

    # State management methods
    async def _load_project_state(self):
        """Load project state from the latest checkpoint"""
        logger.info("Loading project state from checkpoint...")

        checkpoint_dir = Path("persistence/snapshots")
        if not checkpoint_dir.exists():
            logger.warning("No checkpoints directory found")
            return

        # Find latest checkpoint
        checkpoint_files = list(checkpoint_dir.glob(f"checkpoint_*.json"))
        if not checkpoint_files:
            logger.warning("No checkpoint files found")
            return

        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest, 'r') as f:
                checkpoint_data = json.load(f)

            # Restore state
            self.iteration_count = checkpoint_data.get("iteration", 0)
            self.state = ProjectState(checkpoint_data.get("state", "initializing"))

            # Restore metrics
            metrics_data = checkpoint_data.get("metrics", {})
            if metrics_data:
                self.metrics = QualityMetrics(**metrics_data)

            logger.info(f"Loaded state from checkpoint: iteration {self.iteration_count}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    # GEMINI-EDIT - 2025-11-18 - Implemented the _run_tests method to integrate with the Tester Agent.
    async def _run_tests(self) -> Dict:
        """Run the test suite using the TesterAgent and update metrics."""
        logger.info("Executing testing phase...")
        self.state = ProjectState.TESTING
        
        tester_agent = self.agents.get("tester")
        if not tester_agent:
            logger.error("TesterAgent not found. Skipping testing phase.")
            return {"failures": 0, "summary": "Tester agent not available."}

        try:
            task_data = {
                "test_directory": "tests",
                "coverage": True
            }
            
            # The TesterAgent's process_task will call execute_tests
            from .base_agent import AgentTask
            test_task = AgentTask(
                id=f"test-run-{self.iteration_count}",
                type="execute_tests",
                description="Execute full test suite",
                priority=1,
                data=task_data,
                created_at=datetime.now()
            )

            test_run_result = await tester_agent.process_task(test_task)

            if not test_run_result or not test_run_result.get("success"):
                logger.error(f"Test run failed or produced no result. Output: {test_run_result.get('errors')}")
                # Assume critical bugs if the test run itself fails
                self.metrics.bug_count_critical += 1
                return {"failures": 1, "summary": "Test execution failed."}

            # Update metrics from test results
            results = test_run_result.get("results", {})
            passed = results.get("passed", 0)
            failed = results.get("failed", 0)
            
            # For now, we'll classify all failures as critical. A more advanced
            # implementation would involve the AnalyzerAgent classifying bug severity.
            self.metrics.bug_count_critical = failed
            self.metrics.bug_count_minor = 0 # Reset minor bugs for this run
            
            coverage = test_run_result.get("coverage")
            if coverage is not None:
                self.metrics.test_coverage = coverage

            logger.info(f"Test run complete. Passed: {passed}, Failed: {failed}, Coverage: {self.metrics.test_coverage:.2f}%")

            return {
                "failures": failed,
                "summary": f"Passed: {passed}, Failed: {failed}",
                "output": test_run_result.get("output"),
                "errors": test_run_result.get("errors")
            }

        except Exception as e:
            logger.error(f"An exception occurred during the testing phase: {e}")
            self.metrics.bug_count_critical += 1 # Count exception as a critical failure
            return {"failures": 1, "summary": f"Exception during testing: {e}"}
    # GEMINI-EDIT - 2025-11-18 - Implemented the _debug_and_fix method to integrate with the Debugger Agent.
    async def _debug_and_fix(self, test_results: Dict):
        """
        Analyze and attempt to fix test failures using the DebuggerAgent.
        
        Args:
            test_results: The dictionary returned by the _run_tests method.
        """
        logger.info("Executing debugging phase...")
        self.state = ProjectState.DEBUGGING
        
        failures = test_results.get("failures", 0)
        if failures == 0:
            logger.info("No test failures to debug.")
            return

        debugger_agent = self.agents.get("debugger")
        coder_agent = self.agents.get("coder")
        if not debugger_agent or not coder_agent:
            logger.error("DebuggerAgent or CoderAgent not found. Skipping debugging phase.")
            return

        # NOTE: A more robust implementation would parse the test_results['output']
        # to handle each failure individually. For this implementation, we will
        # treat the entire error output as the context for a single debugging task.
        error_output = test_results.get("errors") or test_results.get("output", "")
        
        try:
            # 1. Analyze the error
            logger.info("Dispatching task to DebuggerAgent: analyze_error")
            from .base_agent import AgentTask
            analysis_task = AgentTask(
                id=f"analyze-error-{self.iteration_count}",
                type="analyze_error",
                description="Analyze test failures",
                priority=1,
                data={"error_message": f"{failures} test(s) failed", "stack_trace": error_output},
                created_at=datetime.now()
            )
            analysis_result = await debugger_agent.process_task(analysis_task)

            if not analysis_result or not analysis_result.get("success"):
                logger.error("DebuggerAgent failed to analyze the error.")
                return

            analysis = analysis_result.get("analysis", {})
            error_location = analysis.get("location", {})
            file_path = error_location.get("file")

            if not file_path:
                logger.error("Debugger could not determine the file path of the error.")
                return

            # 2. Read the code from the file where the error occurred
            try:
                with open(file_path, 'r') as f:
                    original_code = f.read()
            except FileNotFoundError:
                logger.error(f"File not found for debugging: {file_path}")
                return

            # 3. Get a fix from the DebuggerAgent
            logger.info("Dispatching task to DebuggerAgent: fix_error")
            fix_task = AgentTask(
                id=f"fix-error-{self.iteration_count}",
                type="fix_error",
                description="Generate a fix for the analyzed error",
                priority=1,
                data={
                    "error_hash": analysis_result.get("error_hash"),
                    "error_message": f"{failures} test(s) failed",
                    "code": original_code,
                    "error_type": analysis.get("error_type")
                },
                created_at=datetime.now()
            )
            fix_result = await debugger_agent.process_task(fix_task)

            if not fix_result or not fix_result.get("success"):
                logger.error("DebuggerAgent failed to generate a fix.")
                return
            
            fixed_code = fix_result.get("fixed_code")

            # 4. Apply the fix using the CoderAgent (or directly write to file)
            logger.info(f"Applying fix to {file_path}...")
            # For simplicity, we write directly. A better approach would be a CoderAgent task.
            try:
                with open(file_path, 'w') as f:
                    f.write(fixed_code)
                logger.success(f"Successfully applied fix to {file_path}")
                # NOTE: A full implementation should now re-run the tests to validate the fix.
                # This would typically involve another call to _run_tests() or looping.
            except Exception as e:
                logger.error(f"Failed to write fix to file {file_path}: {e}")

        except Exception as e:
            logger.error(f"An exception occurred during the debugging phase: {e}")

    async def _optimize_performance(self):
        """Optimize system performance based on metrics"""
        logger.info("Executing performance optimization phase...")
        self.state = ProjectState.OPTIMIZING

        # Check current performance score
        if self.metrics.performance_score >= 90:
            logger.info(f"Performance score {self.metrics.performance_score} already meets target")
            return

        # Generate performance optimization tasks
        if "coder" in self.agents:
            optimization_task = DevelopmentTask(
                id=f"performance-opt-{self.iteration_count}",
                type="optimize_code",
                description="Optimize performance bottlenecks",
                priority=5,
                data={
                    "current_score": self.metrics.performance_score,
                    "target_score": 90,
                    "focus_areas": ["database_queries", "algorithm_efficiency", "caching"]
                }
            )
            self.task_queue.append(optimization_task)
            logger.info("Created performance optimization task")

    async def _validate_quality(self):
        """Validate overall quality against targets"""
        logger.info("Executing quality validation phase...")
        self.state = ProjectState.VALIDATING

        validation_results = {
            "test_coverage": self.metrics.test_coverage >= 95,
            "performance": self.metrics.performance_score >= 90,
            "security": self.metrics.security_score >= 95,
            "critical_bugs": self.metrics.critical_bugs == 0,
            "documentation": self.metrics.documentation_coverage >= 90
        }

        passed = all(validation_results.values())

        if passed:
            logger.success("✓ All quality gates passed!")
        else:
            failed_gates = [k for k, v in validation_results.items() if not v]
            logger.warning(f"Quality gates failed: {', '.join(failed_gates)}")

            # Generate tasks to address failed gates
            for gate in failed_gates:
                if gate == "test_coverage" and "tester" in self.agents:
                    task = DevelopmentTask(
                        id=f"improve-coverage-{self.iteration_count}",
                        type="generate_tests",
                        description="Improve test coverage",
                        priority=2,
                        data={"target_coverage": 95}
                    )
                    self.task_queue.append(task)

        return validation_results
    # GEMINI-EDIT - 2025-11-18 - Implemented the _analyze_current_state method to integrate with the Analyzer Agent.
    async def _analyze_current_state(self) -> Dict:
        """
        Analyze the current state of the project using the AnalyzerAgent.
        """
        logger.info("Executing analysis phase...")
        self.state = ProjectState.PLANNING # Set state to planning as analysis is part of it

        analyzer_agent = self.agents.get("analyzer")
        if not analyzer_agent:
            logger.error("AnalyzerAgent not found. Skipping analysis phase.")
            return {"summary": "Analyzer agent not available."}

        try:
            from .base_agent import AgentTask
            analysis_task = AgentTask(
                id=f"analyze-state-{self.iteration_count}",
                type="monitor_metrics",
                description="Analyze current project metrics",
                priority=1,
                data={
                    "metrics": self.metrics.to_dict(),
                    "iteration": self.iteration_count
                },
                created_at=datetime.now()
            )

            analysis_result = await analyzer_agent.process_task(analysis_task)

            if not analysis_result or not analysis_result.get("success"):
                logger.error("Analysis by AnalyzerAgent failed or produced no result.")
                return {"summary": "Analysis failed."}

            health_status = analysis_result.get("health_status", "unknown")
            alerts = analysis_result.get("alerts", [])
            
            logger.info(f"Project health status: {health_status.upper()}")
            if alerts:
                for alert in alerts:
                    logger.warning(f"Analysis Alert: {alert.get('message')}")

            # The analysis result can be used by the planning method to make more
            # intelligent decisions.
            return analysis_result

        except Exception as e:
            logger.error(f"An exception occurred during the analysis phase: {e}")
            return {"summary": f"Exception during analysis: {e}"}

    async def _recover_from_error(self, error):
        """Attempt to recover from an error"""
        logger.error(f"Attempting error recovery: {error}")

        # Save current state
        await self._create_checkpoint()

        # Analyze error severity
        is_critical = isinstance(error, (SystemExit, KeyboardInterrupt))

        if is_critical:
            logger.critical("Critical error encountered - initiating graceful shutdown")
            await self._cleanup()
            self.stop_requested = True
            return

        # Try to load last known good state
        try:
            await self._load_project_state()
            self.state = ProjectState.PLANNING
            logger.info("Successfully recovered from error - restored to planning state")

        except Exception as recovery_error:
            logger.critical(f"Recovery failed: {recovery_error}")
            # Last resort: reset to initial state
            self.state = ProjectState.INITIALIZING
            self.iteration_count = 0
            logger.warning("Fallback: Reset to initial state")
    async def _cleanup(self): pass
    async def _analyze_successes(self) -> List: return []
    async def _analyze_failures(self) -> List: return []
    async def _learn_from_failure(self, task): pass

    # GEMINI-EDIT - 2025-11-18 - Implemented task generation methods.
    def _generate_bug_fix_tasks(self, priority) -> List[DevelopmentTask]:
        """Generate tasks to fix critical bugs."""
        if self.metrics.bug_count_critical > 0:
            task = DevelopmentTask(
                id=f"fix-bugs-{self.iteration_count}",
                type="debug_code",
                description=f"Fix {self.metrics.bug_count_critical} critical bugs identified in last test run.",
                priority=priority,
                data={"bug_count": self.metrics.bug_count_critical}
            )
            logger.info(f"Generated critical bug fix task: {task.id}")
            return [task]
        return []

    def _generate_test_tasks(self, priority) -> List[DevelopmentTask]:
        """Generate tasks to improve test coverage."""
        if self.metrics.test_coverage < 95.0:
            task = DevelopmentTask(
                id=f"improve-coverage-{self.iteration_count}",
                type="generate_tests",
                description=f"Generate new tests to improve coverage from {self.metrics.test_coverage:.2f}% to 95%.",
                priority=priority,
                data={
                    "current_coverage": self.metrics.test_coverage,
                    "target_coverage": 95.0
                }
            )
            logger.info(f"Generated test generation task: {task.id}")
            return [task]
        return []

    def _generate_optimization_tasks(self, priority) -> List[DevelopmentTask]:
        """Generate tasks to optimize performance."""
        if self.metrics.performance_score < 90.0:
            task = DevelopmentTask(
                id=f"optimize-performance-{self.iteration_count}",
                type="analyze_performance",
                description=f"Analyze and optimize performance bottlenecks. Current score: {self.metrics.performance_score:.2f}%",
                priority=priority,
                data={
                    "current_score": self.metrics.performance_score,
                    "target_score": 90.0
                }
            )
            logger.info(f"Generated performance optimization task: {task.id}")
            return [task]
        return []

    def _generate_feature_tasks(self, priority) -> List[DevelopmentTask]:
        """Generate tasks for implementing new features."""
        # A more sophisticated version would track completed requirements.
        # For now, we'll just create a task for the first requirement if no other tasks exist.
        if not self.task_queue and self.project_spec.get("requirements"):
            next_requirement = self.project_spec["requirements"][0] # Simplified: always takes the first
            task = DevelopmentTask(
                id=f"implement-feature-{self.iteration_count}",
                type="implement_feature",
                description=f"Implement feature: {next_requirement}",
                priority=priority,
                data={
                    "feature_name": next_requirement,
                    "description": next_requirement
                }
            )
            logger.info(f"Generated feature implementation task: {task.id}")
            return [task]
        return []
    def _prioritize_tasks(self, tasks) -> List: return tasks

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

    def _can_stop_safely(self) -> Tuple[bool, str]:
        """
        Enhanced stopping criteria - prevents premature exit.

        Returns:
            (can_stop, reason) tuple
        """
        # CRITICAL: Cannot stop during critical operations
        if self.cannot_stop_flag:
            return False, "Critical operation in progress - cannot stop"

        # Check if user requested stop
        if self.stop_requested:
            return True, "User requested stop"

        # Check if perfection achieved
        if self.metrics.is_perfect():
            # Additional validation: ensure all tasks actually complete
            incomplete_tasks = [
                t for t in self.task_queue
                if t.status not in ['completed', 'cancelled']
            ]

            if incomplete_tasks:
                logger.warning(
                    f"Metrics show perfection but {len(incomplete_tasks)} tasks incomplete. "
                    "Continuing execution."
                )
                return False, f"{len(incomplete_tasks)} tasks remain incomplete"

            # Verify no critical bugs through validation
            if self.metrics.bug_count_critical == 0:
                return True, "All quality metrics achieved and verified"

        # Check for stagnation (no progress)
        if self.stagnant_iterations >= self.max_stagnant_iterations:
            logger.error(
                f"No progress for {self.stagnant_iterations} iterations. "
                "System may be stuck."
            )
            return True, f"Stagnation detected after {self.stagnant_iterations} iterations"

        # Default: keep working
        return False, "Quality criteria not yet met - continuing work"

    def _track_progress(self):
        """
        Track progress to detect stagnation.

        Updates stagnant_iterations counter based on whether meaningful progress occurred.
        """
        current_metrics = {
            'test_coverage': self.metrics.test_coverage,
            'critical_bugs': self.metrics.bug_count_critical,
            'performance_score': self.metrics.performance_score,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks)
        }

        # First iteration - just record
        if self.last_progress_metrics is None:
            self.last_progress_metrics = current_metrics
            self.progress_history.append({
                'iteration': self.iteration_count,
                'metrics': current_metrics,
                'progress': True
            })
            return

        # Check if any metric improved
        progress_made = False

        if current_metrics['test_coverage'] > self.last_progress_metrics['test_coverage']:
            logger.info(
                f"Test coverage improved: "
                f"{self.last_progress_metrics['test_coverage']:.1f}% → "
                f"{current_metrics['test_coverage']:.1f}%"
            )
            progress_made = True

        if current_metrics['critical_bugs'] < self.last_progress_metrics['critical_bugs']:
            logger.info(
                f"Critical bugs reduced: "
                f"{self.last_progress_metrics['critical_bugs']} → "
                f"{current_metrics['critical_bugs']}"
            )
            progress_made = True

        if current_metrics['performance_score'] > self.last_progress_metrics['performance_score']:
            logger.info(
                f"Performance improved: "
                f"{self.last_progress_metrics['performance_score']:.1f}% → "
                f"{current_metrics['performance_score']:.1f}%"
            )
            progress_made = True

        if current_metrics['completed_tasks'] > self.last_progress_metrics['completed_tasks']:
            new_completions = current_metrics['completed_tasks'] - self.last_progress_metrics['completed_tasks']
            logger.info(f"{new_completions} new task(s) completed")
            progress_made = True

        # Update stagnation counter
        if progress_made:
            self.stagnant_iterations = 0
        else:
            self.stagnant_iterations += 1
            logger.warning(
                f"No progress in iteration {self.iteration_count}. "
                f"Stagnant for {self.stagnant_iterations} iterations."
            )

        # Record in history
        self.progress_history.append({
            'iteration': self.iteration_count,
            'metrics': current_metrics,
            'progress': progress_made
        })

        # Update last metrics
        self.last_progress_metrics = current_metrics
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
                priority=9,
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
                priority=8,
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
                priority=10,
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
                priority=6,
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
                priority=9,
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

        return weak_metrics

    async def on_agent_task_complete(self, agent_id: str, task: "AgentTask"):
        """
        Callback invoked by agents when they complete a task.

        Args:
            agent_id: ID of the agent that completed the task
            task: The completed AgentTask
        """
        logger.info(f"Agent {agent_id} completed task {task.id}")

        # Update metrics based on task results
        if task.result:
            # Log to project ledger if available
            if self.project_ledger:
                self.project_ledger.record_decision(
                    iteration=self.iteration_count,
                    agent=agent_id,
                    decision_type="task_completion",
                    description=task.description,
                    rationale=f"Task {task.type} completed",
                    outcome="success" if not task.error else "failure"
                )

        # Check if task triggered any dependent tasks
        # This would be where we implement task dependency management
        pass

    def route_message(self, message: Dict):
        """
        Route inter-agent messages.

        Args:
            message: Message dict with 'from', 'to', 'type', 'content', 'timestamp'
        """
        recipient_id = message.get("to")
        sender_id = message.get("from")

        logger.debug(f"Routing message from {sender_id} to {recipient_id}")

        # Find the recipient agent
        recipient_agent = None
        for agent_name, agent in self.agents.items():
            if agent.id == recipient_id or agent_name == recipient_id:
                recipient_agent = agent
                break

        if recipient_agent:
            # Deliver message to agent
            # For now, agents don't have a message inbox, so we log it
            logger.info(f"Message delivered to {recipient_id}: {message.get('type')}")
            # TODO: Implement actual message delivery when agents support it
        else:
            logger.warning(f"Could not route message - recipient {recipient_id} not found")

# GEMINI-EDIT - 2025-11-18 - Added main execution block to allow the director to be run directly.
if __name__ == "__main__":
    # This allows the director to be run as a standalone script, which is how
    # the start_22_myagent.py launcher executes it.

    # Define a default project for the E2E test: a simple command-line calculator.
    calculator_project_spec = {
        "description": "A simple command-line calculator that can perform addition, subtraction, multiplication, and division.",
        "requirements": [
            "Create a main application file `calculator/main.py`.",
            "Implement a function for each operation: add, subtract, multiply, divide.",
            "Implement a command-line interface to take user input (e.g., '5 + 3').",
            "Handle basic errors like division by zero and invalid input.",
        ],
        "initial_files": {
            "calculator/__init__.py": "",
        }
    }

    async def main():
        """Initializes and starts the director."""
        logger.info("Starting Continuous Director independently for development/testing.")
        
        director = ContinuousDirector(
            project_name="CalculatorCLI",
            project_spec=calculator_project_spec
        )
        
        # In a real scenario, you might load state before starting
        # await director._load_project_state()
        
        await director.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Director stopped by user.")
    except Exception as e:
        logger.critical(f"The director has crashed: {e}", exc_info=True)
