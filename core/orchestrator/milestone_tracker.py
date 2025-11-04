"""
Milestone Tracker - Manages long-term goals and tracks progress towards project completion
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass, field
import json
from pathlib import Path
from loguru import logger


class MilestoneStatus(Enum):
    """Status of a project milestone"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


@dataclass
class Milestone:
    """Represents a project milestone"""
    id: str
    name: str
    description: str
    criteria: Dict  # Specific measurable criteria for completion
    dependencies: List[str] = field(default_factory=list)
    estimated_iterations: int = 10
    actual_iterations: int = 0
    status: MilestoneStatus = MilestoneStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    blocked_reason: Optional[str] = None
    progress_percentage: float = 0.0
    sub_tasks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert milestone to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "criteria": self.criteria,
            "dependencies": self.dependencies,
            "estimated_iterations": self.estimated_iterations,
            "actual_iterations": self.actual_iterations,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "blocked_reason": self.blocked_reason,
            "progress_percentage": self.progress_percentage,
            "sub_tasks": self.sub_tasks
        }


class MilestoneTracker:
    """Tracks and manages project milestones"""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.milestones: Dict[str, Milestone] = {}
        self.milestone_order: List[str] = []  # Ordered list of milestone IDs
        self.current_milestone_id: Optional[str] = None
        self.completed_milestones: List[str] = []
        self.milestone_history: List[Dict] = []

        self._load_milestones()

    def _load_milestones(self):
        """Load existing milestones from storage"""
        milestone_file = Path(f"persistence/milestones_{self.project_name}.json")
        if milestone_file.exists():
            with open(milestone_file, "r") as f:
                data = json.load(f)
                for m_data in data.get("milestones", []):
                    milestone = self._dict_to_milestone(m_data)
                    self.milestones[milestone.id] = milestone
                self.milestone_order = data.get("order", [])
                self.current_milestone_id = data.get("current", None)
                self.completed_milestones = data.get("completed", [])
                self.milestone_history = data.get("history", [])

    def _dict_to_milestone(self, data: Dict) -> Milestone:
        """Convert dictionary to Milestone object"""
        milestone = Milestone(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            criteria=data["criteria"],
            dependencies=data.get("dependencies", []),
            estimated_iterations=data.get("estimated_iterations", 10),
            actual_iterations=data.get("actual_iterations", 0),
            status=MilestoneStatus(data.get("status", "pending")),
            progress_percentage=data.get("progress_percentage", 0.0),
            sub_tasks=data.get("sub_tasks", [])
        )

        if data.get("created_at"):
            milestone.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            milestone.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            milestone.completed_at = datetime.fromisoformat(data["completed_at"])

        milestone.blocked_reason = data.get("blocked_reason")

        return milestone

    def save_milestones(self):
        """Save milestones to persistent storage"""
        milestone_file = Path(f"persistence/milestones_{self.project_name}.json")
        milestone_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "milestones": [m.to_dict() for m in self.milestones.values()],
            "order": self.milestone_order,
            "current": self.current_milestone_id,
            "completed": self.completed_milestones,
            "history": self.milestone_history
        }

        with open(milestone_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_milestone(
        self,
        name: str,
        description: str,
        criteria: Dict,
        dependencies: List[str] = None,
        estimated_iterations: int = 10
    ) -> Milestone:
        """Create a new milestone"""
        milestone_id = f"milestone_{len(self.milestones) + 1}"

        milestone = Milestone(
            id=milestone_id,
            name=name,
            description=description,
            criteria=criteria,
            dependencies=dependencies or [],
            estimated_iterations=estimated_iterations
        )

        self.milestones[milestone_id] = milestone
        self.milestone_order.append(milestone_id)

        logger.info(f"Created milestone: {name} ({milestone_id})")
        self.save_milestones()

        return milestone

    def start_milestone(self, milestone_id: str):
        """Start working on a milestone"""
        if milestone_id not in self.milestones:
            raise ValueError(f"Milestone {milestone_id} not found")

        milestone = self.milestones[milestone_id]

        # Check dependencies
        for dep_id in milestone.dependencies:
            if dep_id not in self.completed_milestones:
                milestone.status = MilestoneStatus.BLOCKED
                milestone.blocked_reason = f"Waiting for dependency: {dep_id}"
                logger.warning(f"Milestone {milestone_id} blocked by {dep_id}")
                return

        milestone.status = MilestoneStatus.IN_PROGRESS
        milestone.started_at = datetime.now()
        self.current_milestone_id = milestone_id

        self.milestone_history.append({
            "timestamp": datetime.now().isoformat(),
            "event": "milestone_started",
            "milestone_id": milestone_id,
            "milestone_name": milestone.name
        })

        logger.info(f"Started milestone: {milestone.name}")
        self.save_milestones()

    def update_progress(self, milestone_id: str, progress_percentage: float, iteration_count: int = 1):
        """Update progress on a milestone"""
        if milestone_id not in self.milestones:
            return

        milestone = self.milestones[milestone_id]
        milestone.progress_percentage = min(100.0, progress_percentage)
        milestone.actual_iterations += iteration_count

        # Check if milestone criteria are met
        if self._check_criteria_met(milestone):
            self.complete_milestone(milestone_id)

        self.save_milestones()

    def _check_criteria_met(self, milestone: Milestone) -> bool:
        """Check if milestone criteria are met"""
        # This would be implemented based on actual project metrics
        # For now, return True if progress is 100%
        return milestone.progress_percentage >= 100.0

    def complete_milestone(self, milestone_id: str):
        """Mark a milestone as completed"""
        if milestone_id not in self.milestones:
            return

        milestone = self.milestones[milestone_id]
        milestone.status = MilestoneStatus.COMPLETED
        milestone.completed_at = datetime.now()
        milestone.progress_percentage = 100.0

        self.completed_milestones.append(milestone_id)

        self.milestone_history.append({
            "timestamp": datetime.now().isoformat(),
            "event": "milestone_completed",
            "milestone_id": milestone_id,
            "milestone_name": milestone.name,
            "actual_iterations": milestone.actual_iterations,
            "estimated_iterations": milestone.estimated_iterations
        })

        logger.success(f"Completed milestone: {milestone.name} in {milestone.actual_iterations} iterations")

        # Check for next milestone
        self._select_next_milestone()

        self.save_milestones()

    def _select_next_milestone(self):
        """Select the next milestone to work on"""
        self.current_milestone_id = None

        for milestone_id in self.milestone_order:
            if milestone_id not in self.completed_milestones:
                milestone = self.milestones[milestone_id]

                # Check if dependencies are met
                deps_met = all(dep in self.completed_milestones for dep in milestone.dependencies)

                if deps_met:
                    self.start_milestone(milestone_id)
                    break

    def get_current_milestone(self) -> Optional[Milestone]:
        """Get the currently active milestone"""
        if self.current_milestone_id:
            return self.milestones.get(self.current_milestone_id)
        return None

    def get_progress_summary(self) -> Dict:
        """Get overall progress summary"""
        total_milestones = len(self.milestones)
        completed = len(self.completed_milestones)

        overall_progress = 0.0
        if total_milestones > 0:
            # Calculate weighted progress
            for milestone in self.milestones.values():
                weight = 1.0 / total_milestones
                overall_progress += milestone.progress_percentage * weight

        return {
            "total_milestones": total_milestones,
            "completed_milestones": completed,
            "current_milestone": self.current_milestone_id,
            "overall_progress": overall_progress,
            "milestones": {
                m_id: {
                    "name": m.name,
                    "status": m.status.value,
                    "progress": m.progress_percentage
                }
                for m_id, m in self.milestones.items()
            }
        }

    def generate_default_milestones(self, project_spec: Dict):
        """Generate default milestones based on project specification"""
        # Basic milestones for any application
        self.create_milestone(
            name="Project Setup",
            description="Initialize project structure and basic configuration",
            criteria={"structure_created": True, "config_complete": True},
            estimated_iterations=5
        )

        self.create_milestone(
            name="Core Functionality",
            description="Implement main features and business logic",
            criteria={"features_implemented": True, "business_logic_complete": True},
            dependencies=["milestone_1"],
            estimated_iterations=20
        )

        self.create_milestone(
            name="Testing Coverage",
            description="Achieve comprehensive test coverage",
            criteria={"test_coverage": 95.0, "all_tests_passing": True},
            dependencies=["milestone_2"],
            estimated_iterations=15
        )

        self.create_milestone(
            name="Performance Optimization",
            description="Optimize for speed and efficiency",
            criteria={"performance_score": 90.0, "load_time_ms": 100},
            dependencies=["milestone_2"],
            estimated_iterations=10
        )

        self.create_milestone(
            name="Production Ready",
            description="Final polish and production deployment",
            criteria={"no_critical_bugs": True, "documentation_complete": True, "deployed": True},
            dependencies=["milestone_3", "milestone_4"],
            estimated_iterations=10
        )

        logger.info(f"Generated {len(self.milestones)} default milestones")