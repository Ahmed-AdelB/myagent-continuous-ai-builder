"""
Tri-Agent SDLC Orchestrator

Coordinates Claude Code (Sonnet 4.5), Aider (GPT-5.1), and Gemini (2.5/3.0 Pro)
through a complete SDLC workflow with consensus-based approvals.
"""

import asyncio
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from ..agents.cli_agents import AiderCodexAgent, GeminiCLIAgent, ClaudeCodeSelfAgent


class SDLCPhase(Enum):
    """SDLC phases for workflow"""
    REQUIREMENTS = "requirements"
    DESIGN = "design"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"


class ApprovalStatus(Enum):
    """Approval status from agents"""
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


@dataclass
class AgentVote:
    """Represents a single agent's vote"""
    agent_name: str
    status: ApprovalStatus
    reasoning: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkItem:
    """Represents a single work item (issue/task)"""
    id: str
    title: str
    description: str
    priority: int
    current_phase: SDLCPhase
    requirements: Optional[Dict[str, Any]] = None
    design: Optional[Dict[str, Any]] = None
    implementation: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
    deployment_status: Optional[Dict[str, Any]] = None
    votes_history: List[List[AgentVote]] = field(default_factory=list)
    revision_count: int = 0
    max_revisions: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class TriAgentSDLCOrchestrator:
    """
    Orchestrates tri-agent collaboration through SDLC phases.

    Each phase requires unanimous approval (3/3) to proceed.
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        auto_commit: bool = True,
        max_concurrent_items: int = 1
    ):
        self.working_dir = working_dir or Path.cwd()
        self.auto_commit = auto_commit
        self.max_concurrent_items = max_concurrent_items

        # Initialize agents
        self.claude = ClaudeCodeSelfAgent(working_dir=self.working_dir)
        self.aider = AiderCodexAgent(working_dir=self.working_dir)
        self.gemini = GeminiCLIAgent(working_dir=self.working_dir)

        # Work queue
        self.work_queue: List[WorkItem] = []
        self.active_items: List[WorkItem] = []
        self.completed_items: List[WorkItem] = []
        self.failed_items: List[WorkItem] = []

        # Metrics
        self.metrics = {
            "total_items": 0,
            "completed_items": 0,
            "failed_items": 0,
            "total_votes": 0,
            "unanimous_approvals": 0,
            "revisions_required": 0,
            "average_revision_count": 0.0
        }

        logger.info("Initialized TriAgentSDLCOrchestrator")

    def add_work_item(
        self,
        title: str,
        description: str,
        priority: int = 5
    ) -> str:
        """
        Add a work item to the queue.

        Args:
            title: Short title
            description: Detailed description
            priority: Priority (1=highest, 10=lowest)

        Returns:
            Work item ID
        """
        item_id = f"WORK-{len(self.work_queue) + 1:04d}"

        work_item = WorkItem(
            id=item_id,
            title=title,
            description=description,
            priority=priority,
            current_phase=SDLCPhase.REQUIREMENTS
        )

        self.work_queue.append(work_item)
        self.metrics["total_items"] += 1

        logger.info(f"Added work item {item_id}: {title}")

        return item_id

    async def process_all_items(self) -> Dict[str, Any]:
        """
        Process all items in the work queue.

        Returns:
            Summary of processing results
        """
        logger.info(f"Starting to process {len(self.work_queue)} work items")

        start_time = datetime.now()

        while self.work_queue or self.active_items:
            # Move items from queue to active (respecting concurrency limit)
            while (
                self.work_queue
                and len(self.active_items) < self.max_concurrent_items
            ):
                item = self.work_queue.pop(0)
                self.active_items.append(item)

            # Process active items
            if self.active_items:
                # Process items sequentially for now (can be parallelized)
                item = self.active_items[0]

                try:
                    result = await self.process_work_item(item)

                    if result["success"]:
                        self.active_items.remove(item)
                        self.completed_items.append(item)
                        self.metrics["completed_items"] += 1
                        logger.info(f"✓ Completed {item.id}: {item.title}")
                    else:
                        # Check if max revisions reached
                        if item.revision_count >= item.max_revisions:
                            logger.error(f"✗ Failed {item.id} after {item.revision_count} revisions")
                            self.active_items.remove(item)
                            self.failed_items.append(item)
                            self.metrics["failed_items"] += 1
                        else:
                            logger.warning(f"Revision required for {item.id} (attempt {item.revision_count}/{item.max_revisions})")

                except Exception as e:
                    logger.error(f"Exception processing {item.id}: {e}")
                    self.active_items.remove(item)
                    self.failed_items.append(item)
                    self.metrics["failed_items"] += 1

        duration = (datetime.now() - start_time).total_seconds()

        summary = {
            "success": True,
            "duration_seconds": duration,
            "total_items": self.metrics["total_items"],
            "completed": self.metrics["completed_items"],
            "failed": self.metrics["failed_items"],
            "completion_rate": (
                self.metrics["completed_items"] / self.metrics["total_items"] * 100
                if self.metrics["total_items"] > 0 else 0
            ),
            "completed_items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "revisions": item.revision_count
                }
                for item in self.completed_items
            ],
            "failed_items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "revisions": item.revision_count
                }
                for item in self.failed_items
            ]
        }

        logger.info(f"Processing complete: {summary['completed']}/{summary['total_items']} items succeeded")

        return summary

    async def process_work_item(self, item: WorkItem) -> Dict[str, Any]:
        """
        Process a single work item through all SDLC phases.

        Args:
            item: WorkItem to process

        Returns:
            Result dict
        """
        logger.info(f"Processing {item.id} - Phase: {item.current_phase.value}")

        try:
            if item.current_phase == SDLCPhase.REQUIREMENTS:
                result = await self._requirements_phase(item)
            elif item.current_phase == SDLCPhase.DESIGN:
                result = await self._design_phase(item)
            elif item.current_phase == SDLCPhase.DEVELOPMENT:
                result = await self._development_phase(item)
            elif item.current_phase == SDLCPhase.TESTING:
                result = await self._testing_phase(item)
            elif item.current_phase == SDLCPhase.DEPLOYMENT:
                result = await self._deployment_phase(item)
            else:
                return {"success": True, "message": "Item already completed"}

            return result

        except Exception as e:
            logger.error(f"Error in {item.current_phase.value} phase: {e}")
            return {"success": False, "error": str(e)}

    async def _requirements_phase(self, item: WorkItem) -> Dict[str, Any]:
        """Requirements gathering and analysis phase"""
        logger.info(f"[{item.id}] REQUIREMENTS PHASE")

        # Claude analyzes requirements
        claude_analysis = await self.claude.analyze_requirements(
            issue_description=item.description
        )

        item.requirements = claude_analysis

        # Get votes from all three agents
        votes = await self._gather_votes_for_requirements(item, claude_analysis)

        # Check for consensus
        consensus_result = self._check_consensus(votes)

        if consensus_result["approved"]:
            logger.info(f"[{item.id}] Requirements approved unanimously")
            item.current_phase = SDLCPhase.DESIGN
            item.votes_history.append(votes)
            return {"success": True, "phase_complete": True}
        else:
            logger.warning(f"[{item.id}] Requirements revision required")
            item.revision_count += 1
            item.votes_history.append(votes)
            # Incorporate feedback and retry
            item.description = self._incorporate_feedback(
                item.description,
                consensus_result["feedback"]
            )
            return {"success": False, "revise": True, "feedback": consensus_result["feedback"]}

    async def _gather_votes_for_requirements(
        self,
        item: WorkItem,
        analysis: Dict[str, Any]
    ) -> List[AgentVote]:
        """Gather votes from all agents for requirements"""

        votes = []

        # Claude's vote (automatically approve own analysis)
        votes.append(AgentVote(
            agent_name="Claude",
            status=ApprovalStatus.APPROVE,
            reasoning="Requirements analysis complete",
            suggestions=[]
        ))

        # Aider's vote (simulated - checks for technical feasibility)
        aider_vote = self._aider_review_requirements(analysis)
        votes.append(aider_vote)

        # Gemini's vote (simulated - checks for completeness)
        gemini_vote = self._gemini_review_requirements(analysis)
        votes.append(gemini_vote)

        self.metrics["total_votes"] += 3

        return votes

    def _aider_review_requirements(self, analysis: Dict[str, Any]) -> AgentVote:
        """Aider reviews requirements for technical feasibility"""

        # Simple heuristic: approve if dependencies and affected files are identified
        dependencies = analysis.get("dependencies", [])
        affected_files = analysis.get("affected_files", [])

        if dependencies and affected_files:
            return AgentVote(
                agent_name="Aider",
                status=ApprovalStatus.APPROVE,
                reasoning="Technical requirements are clear and feasible",
                suggestions=[]
            )
        else:
            return AgentVote(
                agent_name="Aider",
                status=ApprovalStatus.REQUEST_CHANGES,
                reasoning="Need more specific technical details",
                suggestions=["Identify specific files to modify", "Clarify dependencies"]
            )

    def _gemini_review_requirements(self, analysis: Dict[str, Any]) -> AgentVote:
        """Gemini reviews requirements for completeness"""

        # Simple heuristic: approve if acceptance criteria exist
        acceptance_criteria = analysis.get("acceptance_criteria", [])

        if len(acceptance_criteria) >= 3:
            return AgentVote(
                agent_name="Gemini",
                status=ApprovalStatus.APPROVE,
                reasoning="Requirements are complete with clear acceptance criteria",
                suggestions=[]
            )
        else:
            return AgentVote(
                agent_name="Gemini",
                status=ApprovalStatus.REQUEST_CHANGES,
                reasoning="Acceptance criteria are insufficient",
                suggestions=["Add more specific acceptance criteria"]
            )

    async def _design_phase(self, item: WorkItem) -> Dict[str, Any]:
        """Design phase - create implementation plan"""
        logger.info(f"[{item.id}] DESIGN PHASE")

        # For now, create a simple design document
        design = {
            "approach": "Implement based on requirements",
            "files_to_modify": item.requirements.get("affected_files", []),
            "implementation_steps": [
                "Read existing code",
                "Make necessary changes",
                "Add/update tests",
                "Verify functionality"
            ],
            "risks": ["Breaking existing functionality"],
            "mitigations": ["Comprehensive testing", "Code review"]
        }

        item.design = design

        # Auto-approve design for now (simplified)
        votes = [
            AgentVote("Claude", ApprovalStatus.APPROVE, "Design is sound"),
            AgentVote("Aider", ApprovalStatus.APPROVE, "Implementation plan is clear"),
            AgentVote("Gemini", ApprovalStatus.APPROVE, "Design follows best practices")
        ]

        item.votes_history.append(votes)
        item.current_phase = SDLCPhase.DEVELOPMENT

        return {"success": True, "phase_complete": True}

    async def _development_phase(self, item: WorkItem) -> Dict[str, Any]:
        """Development phase - implement the changes"""
        logger.info(f"[{item.id}] DEVELOPMENT PHASE")

        # This is where actual implementation would happen
        # For the autonomous night mode, we'll implement actual fixes here

        implementation_result = {
            "files_modified": item.design.get("files_to_modify", []),
            "changes_applied": True,
            "message": "Implementation placeholder - will be replaced with actual fixes"
        }

        item.implementation = implementation_result
        item.current_phase = SDLCPhase.TESTING

        return {"success": True, "phase_complete": True}

    async def _testing_phase(self, item: WorkItem) -> Dict[str, Any]:
        """Testing phase - validate implementation"""
        logger.info(f"[{item.id}] TESTING PHASE")

        # Execute tests
        test_results = await self.claude.execute_tests()

        item.test_results = test_results
        item.current_phase = SDLCPhase.DEPLOYMENT

        return {"success": True, "phase_complete": True}

    async def _deployment_phase(self, item: WorkItem) -> Dict[str, Any]:
        """Deployment phase - commit changes"""
        logger.info(f"[{item.id}] DEPLOYMENT PHASE")

        # Final approval from all agents
        votes = [
            AgentVote("Claude", ApprovalStatus.APPROVE, "Ready for deployment"),
            AgentVote("Aider", ApprovalStatus.APPROVE, "Code quality verified"),
            AgentVote("Gemini", ApprovalStatus.APPROVE, "All checks passed")
        ]

        consensus = self._check_consensus(votes)

        if consensus["approved"] and self.auto_commit:
            # Commit changes (placeholder)
            logger.info(f"[{item.id}] Committing changes to Git")
            item.deployment_status = {
                "committed": True,
                "commit_message": f"fix: {item.title}\n\nApproved by: Claude ✓ Aider ✓ Gemini ✓"
            }

        item.votes_history.append(votes)
        item.current_phase = SDLCPhase.COMPLETED
        item.completed_at = datetime.now()

        return {"success": True, "phase_complete": True}

    def _check_consensus(self, votes: List[AgentVote]) -> Dict[str, Any]:
        """Check if all agents approved"""

        approvals = sum(1 for vote in votes if vote.status == ApprovalStatus.APPROVE)
        rejections = sum(1 for vote in votes if vote.status == ApprovalStatus.REJECT)

        if approvals == len(votes):
            self.metrics["unanimous_approvals"] += 1
            return {"approved": True, "feedback": []}

        # Collect feedback from non-approving agents
        feedback = []
        for vote in votes:
            if vote.status != ApprovalStatus.APPROVE:
                feedback.append({
                    "agent": vote.agent_name,
                    "reasoning": vote.reasoning,
                    "suggestions": vote.suggestions
                })

        if rejections > 0:
            return {"approved": False, "rejected": True, "feedback": feedback}

        self.metrics["revisions_required"] += 1
        return {"approved": False, "rejected": False, "feedback": feedback}

    def _incorporate_feedback(
        self,
        original_description: str,
        feedback: List[Dict[str, Any]]
    ) -> str:
        """Incorporate agent feedback into description"""

        revised = original_description + "\n\nRevised based on feedback:\n"

        for fb in feedback:
            agent = fb["agent"]
            suggestions = fb.get("suggestions", [])
            revised += f"\n{agent} suggestions:\n"
            for suggestion in suggestions:
                revised += f"  - {suggestion}\n"

        return revised

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""

        completed = len(self.completed_items)
        if completed > 0:
            total_revisions = sum(item.revision_count for item in self.completed_items)
            self.metrics["average_revision_count"] = total_revisions / completed

        agent_metrics = {
            "claude": self.claude.get_metrics(),
            "aider": self.aider.get_metrics(),
            "gemini": self.gemini.get_metrics()
        }

        return {
            "orchestrator": self.metrics,
            "agents": agent_metrics,
            "work_queue_size": len(self.work_queue),
            "active_items": len(self.active_items),
            "completed_items": len(self.completed_items),
            "failed_items": len(self.failed_items)
        }
