"""
Base Agent - Foundation class for all persistent AI agents
"""

import json
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import uuid
from enum import Enum
from loguru import logger
import pickle
from pathlib import Path


class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    WORKING = "working"
    THINKING = "thinking"
    WAITING = "waiting"
    ERROR = "error"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class AgentMemory:
    """Agent's working memory"""
    short_term: List[Dict] = field(default_factory=list)
    working_context: Dict = field(default_factory=dict)
    learned_patterns: List[Dict] = field(default_factory=list)
    decision_history: List[Dict] = field(default_factory=list)
    error_encounters: List[Dict] = field(default_factory=list)


@dataclass
class AgentTask:
    """Represents a task for an agent"""
    id: str
    type: str
    description: str
    priority: int
    data: Dict
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3


class PersistentAgent(ABC):
    """Base class for all persistent agents in the system"""
    
    def __init__(
        self,
        name: str,
        role: str,
        capabilities: List[str],
        orchestrator=None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.orchestrator = orchestrator
        
        # Agent state
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.current_task: Optional[AgentTask] = None
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0,
            "average_task_time": 0,
            "success_rate": 0,
            "error_recovery_rate": 0
        }
        
        # Checkpoint support
        self.checkpoint_dir = Path(f"persistence/agents/{self.name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint: Optional[datetime] = None
        
        # Communication
        self.message_handlers: Dict[str, Callable] = {}
        self.outgoing_messages: List[Dict] = []
        
        # Learning
        self.learning_enabled = True
        self.adaptation_threshold = 0.8
        
        logger.info(f"Initialized {self.role} agent: {self.name} ({self.id})")
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Any:
        """Process a specific task - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def analyze_context(self, context: Dict) -> Dict:
        """Analyze current context - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def generate_solution(self, problem: Dict) -> Dict:
        """Generate solution for a problem - must be implemented by subclasses"""
        pass
    
    async def start(self):
        """Start the agent's main loop"""
        logger.info(f"Starting agent {self.name}")
        self.state = AgentState.IDLE
        
        # Load checkpoint if exists
        self.load_checkpoint()
        
        # Main agent loop
        while self.state != AgentState.COMPLETED:
            try:
                if self.state == AgentState.ERROR:
                    await self.recover_from_error()
                
                if self.task_queue and self.state == AgentState.IDLE:
                    await self.execute_next_task()
                
                # Process messages
                await self.process_messages()
                
                # Periodic checkpoint
                if self.should_checkpoint():
                    self.save_checkpoint()
                
                await asyncio.sleep(0.001)  # Minimal 1ms yield for safety-critical responsiveness
                
            except Exception as e:
                logger.error(f"Error in agent {self.name} main loop: {e}")
                self.state = AgentState.ERROR
                self.memory.error_encounters.append({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "context": self.memory.working_context
                })
    
    async def execute_next_task(self):
        """Execute the next task in queue"""
        if not self.task_queue:
            return
        
        self.current_task = self.task_queue.pop(0)
        self.current_task.started_at = datetime.now()
        self.current_task.attempts += 1
        
        self.state = AgentState.WORKING
        logger.info(f"Agent {self.name} starting task: {self.current_task.description}")
        
        try:
            # Update working context
            self.memory.working_context = {
                "task_id": self.current_task.id,
                "task_type": self.current_task.type,
                "iteration": self.orchestrator.iteration_count if self.orchestrator else 0
            }
            
            # Process the task
            result = await self.process_task(self.current_task)
            
            # Task completed successfully
            self.current_task.result = result
            self.current_task.completed_at = datetime.now()
            
            self.completed_tasks.append(self.current_task)
            self.metrics["tasks_completed"] += 1
            
            # Learn from success
            if self.learning_enabled:
                self.learn_from_task(self.current_task, success=True)
            
            # Notify orchestrator
            if self.orchestrator:
                await self.orchestrator.on_agent_task_complete(
                    self.id,
                    self.current_task
                )
            
            logger.info(f"Agent {self.name} completed task: {self.current_task.id}")
            
        except Exception as e:
            logger.error(f"Agent {self.name} failed task: {e}")
            self.current_task.error = str(e)
            
            if self.current_task.attempts < self.current_task.max_attempts:
                # Retry the task
                self.task_queue.insert(0, self.current_task)
                logger.info(f"Retrying task {self.current_task.id} "
                          f"(attempt {self.current_task.attempts}/{self.current_task.max_attempts})")
            else:
                # Task failed permanently
                self.current_task.completed_at = datetime.now()
                self.completed_tasks.append(self.current_task)
                self.metrics["tasks_failed"] += 1
                
                # Learn from failure
                if self.learning_enabled:
                    self.learn_from_task(self.current_task, success=False)
        
        finally:
            self.current_task = None
            self.state = AgentState.IDLE
            self.update_metrics()
    
    def add_task(self, task: AgentTask):
        """Add a task to the agent's queue"""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        logger.info(f"Added task to {self.name}: {task.description}")
    
    def learn_from_task(self, task: AgentTask, success: bool):
        """Learn from task execution"""
        pattern = {
            "task_type": task.type,
            "success": success,
            "data_characteristics": self._extract_data_characteristics(task.data),
            "execution_time": (
                task.completed_at - task.started_at
            ).total_seconds() if task.completed_at and task.started_at else 0,
            "error": task.error if not success else None
        }
        
        self.memory.learned_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.memory.learned_patterns) > 100:
            self.memory.learned_patterns = self.memory.learned_patterns[-100:]
    
    def _extract_data_characteristics(self, data: Dict) -> Dict:
        """Extract characteristics from task data for learning"""
        return {
            "size": len(str(data)),
            "keys": list(data.keys()) if isinstance(data, dict) else [],
            "types": [type(v).__name__ for v in data.values()] if isinstance(data, dict) else []
        }
    
    async def recover_from_error(self):
        """Attempt to recover from error state"""
        logger.info(f"Agent {self.name} attempting error recovery")
        
        # Clear current task if it exists
        if self.current_task:
            self.current_task.error = "Agent error recovery"
            self.completed_tasks.append(self.current_task)
            self.current_task = None
        
        # Reset memory if needed
        if len(self.memory.error_encounters) > 10:
            self.memory.short_term.clear()
            self.memory.working_context.clear()
        
        # Try to restore from checkpoint
        if self.load_checkpoint():
            logger.info(f"Agent {self.name} recovered from checkpoint")
            self.state = AgentState.IDLE
        else:
            # Reset to clean state
            self.memory = AgentMemory()
            self.state = AgentState.IDLE
            logger.info(f"Agent {self.name} reset to clean state")
    
    def save_checkpoint(self):
        """Save agent state to checkpoint"""
        checkpoint_data = {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "state": self.state.value,
            "memory": {
                "short_term": self.memory.short_term,
                "working_context": self.memory.working_context,
                "learned_patterns": self.memory.learned_patterns[-50:],  # Keep recent
                "decision_history": self.memory.decision_history[-50:]
            },
            "metrics": self.metrics,
            "task_queue": [
                {
                    "id": t.id,
                    "type": t.type,
                    "description": t.description,
                    "priority": t.priority,
                    "data": t.data
                }
                for t in self.task_queue
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save as JSON for readability
        json_file = checkpoint_file.with_suffix('.json')
        with open(json_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        self.last_checkpoint = datetime.now()
        logger.info(f"Saved checkpoint for agent {self.name}")
    
    def load_checkpoint(self) -> bool:
        """Load agent state from checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        if not checkpoints:
            return False
        
        # Get most recent checkpoint
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, "rb") as f:
                data = pickle.load(f)
            
            # Restore state
            self.id = data["id"]
            self.state = AgentState(data["state"])
            self.metrics = data["metrics"]
            
            # Restore memory
            self.memory.short_term = data["memory"]["short_term"]
            self.memory.working_context = data["memory"]["working_context"]
            self.memory.learned_patterns = data["memory"]["learned_patterns"]
            self.memory.decision_history = data["memory"]["decision_history"]
            
            # Restore task queue
            self.task_queue = []
            for t_data in data["task_queue"]:
                task = AgentTask(
                    id=t_data["id"],
                    type=t_data["type"],
                    description=t_data["description"],
                    priority=t_data["priority"],
                    data=t_data["data"],
                    created_at=datetime.now()
                )
                self.task_queue.append(task)
            
            logger.info(f"Loaded checkpoint for agent {self.name} from {latest_checkpoint}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def should_checkpoint(self) -> bool:
        """Determine if a checkpoint should be created"""
        if not self.last_checkpoint:
            return True
        
        time_since_checkpoint = (datetime.now() - self.last_checkpoint).total_seconds()
        
        # Checkpoint every 10 minutes or after 10 completed tasks
        return (
            time_since_checkpoint > 600 or
            len(self.completed_tasks) % 10 == 0
        )
    
    def update_metrics(self):
        """Update agent performance metrics"""
        if self.completed_tasks:
            total_time = sum(
                (t.completed_at - t.started_at).total_seconds()
                for t in self.completed_tasks
                if t.completed_at and t.started_at
            )
            
            self.metrics["total_execution_time"] = total_time
            self.metrics["average_task_time"] = total_time / len(self.completed_tasks)
            
            successful = sum(1 for t in self.completed_tasks if not t.error)
            self.metrics["success_rate"] = successful / len(self.completed_tasks)
    
    async def process_messages(self):
        """Process incoming messages from other agents"""
        # This would be implemented with actual message queue
        pass
    
    def send_message(self, recipient: str, message_type: str, content: Dict):
        """Send message to another agent"""
        message = {
            "from": self.id,
            "to": recipient,
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.outgoing_messages.append(message)
        
        if self.orchestrator:
            self.orchestrator.route_message(message)
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "state": self.state.value,
            "current_task": self.current_task.id if self.current_task else None,
            "queue_size": len(self.task_queue),
            "completed_tasks": self.metrics["tasks_completed"],
            "failed_tasks": self.metrics["tasks_failed"],
            "success_rate": self.metrics["success_rate"],
            "memory_size": len(self.memory.short_term)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        logger.info(f"Shutting down agent {self.name}")
        
        # Save final checkpoint
        self.save_checkpoint()
        
        # Complete current task if any
        if self.current_task:
            self.current_task.error = "Agent shutdown"
            self.completed_tasks.append(self.current_task)
        
        self.state = AgentState.COMPLETED