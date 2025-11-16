"""
Human-in-the-Loop Review Gateway - GPT-5 Recommendation #5
Quality Control Checkpoints with Human Validation

Balances autonomy with governance by inserting review checkpoints
where the system requests human validation for major design or dependency changes.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import aiohttp
from pathlib import Path

from core.communication.agent_message_bus import AgentMessageBus, MessageType, MessagePriority

logger = logging.getLogger(__name__)


class ReviewType(Enum):
    """Types of reviews that can be requested"""
    ARCHITECTURE_CHANGE = "architecture_change"
    DEPENDENCY_UPDATE = "dependency_update"
    MAJOR_REFACTOR = "major_refactor"
    API_BREAKING_CHANGE = "api_breaking_change"
    SECURITY_CHANGE = "security_change"
    PERFORMANCE_CRITICAL = "performance_critical"
    DATA_MODEL_CHANGE = "data_model_change"
    INTEGRATION_CHANGE = "integration_change"


class ReviewStatus(Enum):
    """Status of a review request"""
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class ReviewPriority(Enum):
    """Priority levels for review requests"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ReviewRequest:
    """Human review request with all necessary context"""
    id: str
    review_type: ReviewType
    priority: ReviewPriority
    title: str
    description: str
    impact_assessment: str
    proposed_changes: List[Dict[str, Any]]
    risk_analysis: Dict[str, Any]
    requester_agent: str
    iteration_id: int

    # Review context
    current_state: Dict[str, Any]
    proposed_state: Dict[str, Any]
    rollback_plan: str
    testing_strategy: str

    # Metadata
    created_at: datetime
    expires_at: datetime
    status: ReviewStatus

    # Review outcome
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    decision: Optional[str] = None
    feedback: Optional[str] = None
    conditions: Optional[List[str]] = None


@dataclass
class ReviewConfiguration:
    """Configuration for review requirements"""
    auto_approve_low_risk: bool = True
    require_review_threshold: float = 0.7  # Risk score threshold
    default_expiry_hours: int = 24
    escalation_hours: int = 48

    # Review triggers
    always_require_review: List[ReviewType] = None
    never_require_review: List[ReviewType] = None

    # Notification settings
    slack_webhook: Optional[str] = None
    email_notifications: bool = True
    github_integration: bool = True


class HumanReviewGateway:
    """
    Human-in-the-Loop Review Gateway

    Implements GPT-5 recommendation for balancing autonomy with governance
    by inserting strategic human review checkpoints.
    """

    def __init__(
        self,
        project_name: str,
        message_bus: AgentMessageBus,
        config: Optional[ReviewConfiguration] = None
    ):
        self.project_name = project_name
        self.message_bus = message_bus
        self.config = config or ReviewConfiguration()

        # Review storage
        self.pending_reviews: Dict[str, ReviewRequest] = {}
        self.review_history: List[ReviewRequest] = []

        # Review handlers
        self.review_handlers: Dict[ReviewType, Callable] = {}

        # Persistence
        self.storage_path = Path(f"persistence/reviews_{project_name}")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Human Review Gateway initialized for project: {project_name}")

    async def start(self):
        """Start the review gateway and subscribe to relevant messages"""
        # Subscribe to relevant message types
        await self.message_bus.subscribe(
            agent_id="human_review_gateway",
            message_types=[
                MessageType.DESIGN_REVIEW,
                MessageType.ARCHITECTURE_UPDATE,
                MessageType.CODE_CHANGE
            ],
            callback=self._handle_message
        )

        # Start background tasks
        asyncio.create_task(self._review_expiry_monitor())
        asyncio.create_task(self._load_persisted_reviews())

        logger.info("Human Review Gateway started")

    async def request_review(
        self,
        review_type: ReviewType,
        title: str,
        description: str,
        proposed_changes: List[Dict[str, Any]],
        requester_agent: str,
        iteration_id: int,
        priority: ReviewPriority = ReviewPriority.NORMAL,
        impact_assessment: str = "",
        risk_analysis: Dict[str, Any] = None,
        current_state: Dict[str, Any] = None,
        proposed_state: Dict[str, Any] = None,
        rollback_plan: str = "",
        testing_strategy: str = ""
    ) -> str:
        """
        Request human review for a proposed change

        Args:
            review_type: Type of review needed
            title: Short title describing the change
            description: Detailed description of the change
            proposed_changes: List of specific changes to be made
            requester_agent: Agent requesting the review
            iteration_id: Current iteration ID
            priority: Priority level for the review
            impact_assessment: Assessment of change impact
            risk_analysis: Risk analysis data
            current_state: Current system state
            proposed_state: Proposed system state
            rollback_plan: Plan for rolling back changes if needed
            testing_strategy: Strategy for testing the changes

        Returns:
            str: Review request ID
        """
        review_id = str(uuid.uuid4())

        # Calculate expiry time
        expiry_hours = self.config.default_expiry_hours
        if priority == ReviewPriority.CRITICAL:
            expiry_hours = 4
        elif priority == ReviewPriority.EMERGENCY:
            expiry_hours = 1

        expires_at = datetime.now() + timedelta(hours=expiry_hours)

        # Create review request
        review_request = ReviewRequest(
            id=review_id,
            review_type=review_type,
            priority=priority,
            title=title,
            description=description,
            impact_assessment=impact_assessment,
            proposed_changes=proposed_changes,
            risk_analysis=risk_analysis or {},
            requester_agent=requester_agent,
            iteration_id=iteration_id,
            current_state=current_state or {},
            proposed_state=proposed_state or {},
            rollback_plan=rollback_plan,
            testing_strategy=testing_strategy,
            created_at=datetime.now(),
            expires_at=expires_at,
            status=ReviewStatus.PENDING
        )

        # Check if review is actually needed
        if await self._should_auto_approve(review_request):
            review_request.status = ReviewStatus.APPROVED
            review_request.decision = "auto_approved"
            review_request.feedback = "Automatically approved based on low risk assessment"
            self.review_history.append(review_request)

            logger.info(f"Auto-approved review {review_id} for {review_type.value}")
            return review_id

        # Store pending review
        self.pending_reviews[review_id] = review_request

        # Persist review
        await self._persist_review(review_request)

        # Send notifications
        await self._send_review_notification(review_request)

        # Create GitHub PR if configured
        if self.config.github_integration:
            await self._create_github_pr(review_request)

        logger.info(f"Created review request {review_id} for {review_type.value}")
        return review_id

    async def get_review_status(self, review_id: str) -> Optional[ReviewRequest]:
        """Get the status of a review request"""
        if review_id in self.pending_reviews:
            return self.pending_reviews[review_id]

        # Check history
        for review in self.review_history:
            if review.id == review_id:
                return review

        return None

    async def submit_review_decision(
        self,
        review_id: str,
        reviewer: str,
        decision: str,  # approved, rejected, needs_changes
        feedback: str = "",
        conditions: List[str] = None
    ) -> bool:
        """
        Submit a human review decision

        Args:
            review_id: ID of the review being decided
            reviewer: Name/ID of the reviewer
            decision: Review decision (approved, rejected, needs_changes)
            feedback: Reviewer feedback
            conditions: Conditions that must be met for approval

        Returns:
            bool: True if decision was recorded successfully
        """
        if review_id not in self.pending_reviews:
            logger.error(f"Review {review_id} not found in pending reviews")
            return False

        review = self.pending_reviews[review_id]

        # Update review with decision
        review.reviewer = reviewer
        review.reviewed_at = datetime.now()
        review.decision = decision
        review.feedback = feedback
        review.conditions = conditions or []

        # Update status
        if decision == "approved":
            review.status = ReviewStatus.APPROVED
        elif decision == "rejected":
            review.status = ReviewStatus.REJECTED
        elif decision == "needs_changes":
            review.status = ReviewStatus.NEEDS_CHANGES
        else:
            logger.error(f"Invalid decision: {decision}")
            return False

        # Move to history
        self.review_history.append(review)
        del self.pending_reviews[review_id]

        # Notify requester agent
        await self._notify_review_decision(review)

        # Persist decision
        await self._persist_review(review)

        logger.info(f"Review {review_id} decided: {decision} by {reviewer}")
        return True

    async def wait_for_review(self, review_id: str, timeout_seconds: int = None) -> ReviewRequest:
        """
        Wait for a review to be completed

        Args:
            review_id: Review ID to wait for
            timeout_seconds: Maximum time to wait (None for no timeout)

        Returns:
            ReviewRequest with final status
        """
        start_time = datetime.now()

        while True:
            review = await self.get_review_status(review_id)

            if not review:
                raise ValueError(f"Review {review_id} not found")

            if review.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED,
                               ReviewStatus.NEEDS_CHANGES, ReviewStatus.EXPIRED]:
                return review

            # Check timeout
            if timeout_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Review {review_id} timeout after {timeout_seconds}s")

            await asyncio.sleep(5)  # Check every 5 seconds

    async def _should_auto_approve(self, review_request: ReviewRequest) -> bool:
        """Determine if a review should be automatically approved"""

        # Check configuration
        if review_request.review_type in (self.config.always_require_review or []):
            return False

        if review_request.review_type in (self.config.never_require_review or []):
            return True

        # Check priority
        if review_request.priority in [ReviewPriority.CRITICAL, ReviewPriority.EMERGENCY]:
            return False

        # Check risk score
        risk_score = review_request.risk_analysis.get("overall_score", 1.0)
        if risk_score >= self.config.require_review_threshold:
            return False

        # Auto-approve low-risk changes if configured
        if self.config.auto_approve_low_risk and risk_score < 0.3:
            return True

        return False

    async def _send_review_notification(self, review_request: ReviewRequest):
        """Send review notification via configured channels"""

        notification_data = {
            "review_id": review_request.id,
            "type": review_request.review_type.value,
            "priority": review_request.priority.value,
            "title": review_request.title,
            "requester": review_request.requester_agent,
            "expires_at": review_request.expires_at.isoformat()
        }

        # Send Slack notification
        if self.config.slack_webhook:
            await self._send_slack_notification(notification_data, review_request)

        # Send email notification
        if self.config.email_notifications:
            await self._send_email_notification(notification_data, review_request)

        # Send message bus notification
        await self.message_bus.send_message(
            sender_agent="human_review_gateway",
            message_type=MessageType.CUSTOM,
            payload={
                "event": "review_requested",
                "review_data": notification_data
            },
            priority=MessagePriority.HIGH
        )

    async def _send_slack_notification(self, notification_data: Dict, review_request: ReviewRequest):
        """Send Slack notification for review request"""
        if not self.config.slack_webhook:
            return

        try:
            slack_message = {
                "text": f"🔍 Review Request: {review_request.title}",
                "attachments": [
                    {
                        "color": "warning" if review_request.priority.value in ["high", "critical"] else "good",
                        "fields": [
                            {"title": "Type", "value": review_request.review_type.value, "short": True},
                            {"title": "Priority", "value": review_request.priority.value, "short": True},
                            {"title": "Requester", "value": review_request.requester_agent, "short": True},
                            {"title": "Expires", "value": review_request.expires_at.strftime("%Y-%m-%d %H:%M"), "short": True},
                            {"title": "Description", "value": review_request.description[:500], "short": False}
                        ]
                    }
                ]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.slack_webhook, json=slack_message) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send Slack notification: {response.status}")

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

    async def _create_github_pr(self, review_request: ReviewRequest):
        """Create GitHub PR for review if configured"""
        # This would integrate with GitHub API to create a PR
        # Implementation depends on GitHub configuration
        logger.info(f"GitHub PR creation for review {review_request.id} (placeholder)")

    async def _notify_review_decision(self, review: ReviewRequest):
        """Notify the requesting agent of the review decision"""
        await self.message_bus.send_message(
            sender_agent="human_review_gateway",
            recipient_agent=review.requester_agent,
            message_type=MessageType.DESIGN_REVIEW,
            payload={
                "event": "review_completed",
                "review_id": review.id,
                "status": review.status.value,
                "decision": review.decision,
                "feedback": review.feedback,
                "conditions": review.conditions
            },
            priority=MessagePriority.HIGH
        )

    async def _handle_message(self, message):
        """Handle incoming messages that might trigger reviews"""
        if message.message_type == MessageType.ARCHITECTURE_UPDATE:
            # Check if this architectural change needs review
            risk_score = message.payload.get("risk_score", 0.5)
            if risk_score > 0.6:
                await self.request_review(
                    review_type=ReviewType.ARCHITECTURE_CHANGE,
                    title=f"Architecture Update: {message.payload.get('title', 'Unknown')}",
                    description=message.payload.get("description", ""),
                    proposed_changes=[message.payload],
                    requester_agent=message.sender_agent,
                    iteration_id=message.payload.get("iteration_id", 0),
                    risk_analysis={"overall_score": risk_score}
                )

    async def _review_expiry_monitor(self):
        """Background task to monitor and handle review expiries"""
        while True:
            try:
                current_time = datetime.now()
                expired_reviews = []

                for review_id, review in self.pending_reviews.items():
                    if current_time > review.expires_at:
                        expired_reviews.append(review_id)

                # Handle expired reviews
                for review_id in expired_reviews:
                    review = self.pending_reviews[review_id]
                    review.status = ReviewStatus.EXPIRED
                    review.decision = "expired"
                    review.feedback = "Review expired without decision"

                    # Move to history
                    self.review_history.append(review)
                    del self.pending_reviews[review_id]

                    # Notify requester
                    await self._notify_review_decision(review)

                    logger.warning(f"Review {review_id} expired")

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in review expiry monitor: {e}")
                await asyncio.sleep(60)

    async def _persist_review(self, review: ReviewRequest):
        """Persist review to storage"""
        try:
            review_file = self.storage_path / f"review_{review.id}.json"
            with open(review_file, 'w') as f:
                # Custom serialization for datetime and enums
                review_dict = asdict(review)
                review_dict["created_at"] = review.created_at.isoformat()
                review_dict["expires_at"] = review.expires_at.isoformat()
                if review.reviewed_at:
                    review_dict["reviewed_at"] = review.reviewed_at.isoformat()
                review_dict["review_type"] = review.review_type.value
                review_dict["priority"] = review.priority.value
                review_dict["status"] = review.status.value

                json.dump(review_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist review {review.id}: {e}")

    async def _load_persisted_reviews(self):
        """Load persisted reviews on startup"""
        try:
            for review_file in self.storage_path.glob("review_*.json"):
                with open(review_file, 'r') as f:
                    review_dict = json.load(f)

                    # Reconstruct datetime and enum objects
                    review_dict["created_at"] = datetime.fromisoformat(review_dict["created_at"])
                    review_dict["expires_at"] = datetime.fromisoformat(review_dict["expires_at"])
                    if review_dict.get("reviewed_at"):
                        review_dict["reviewed_at"] = datetime.fromisoformat(review_dict["reviewed_at"])

                    review_dict["review_type"] = ReviewType(review_dict["review_type"])
                    review_dict["priority"] = ReviewPriority(review_dict["priority"])
                    review_dict["status"] = ReviewStatus(review_dict["status"])

                    review = ReviewRequest(**review_dict)

                    # Add to appropriate collection
                    if review.status == ReviewStatus.PENDING:
                        self.pending_reviews[review.id] = review
                    else:
                        self.review_history.append(review)

            logger.info(f"Loaded {len(self.pending_reviews)} pending reviews and {len(self.review_history)} historical reviews")

        except Exception as e:
            logger.error(f"Failed to load persisted reviews: {e}")

    async def get_review_dashboard_data(self) -> Dict[str, Any]:
        """Get data for review dashboard"""
        pending_by_priority = {}
        for priority in ReviewPriority:
            pending_by_priority[priority.value] = len([
                r for r in self.pending_reviews.values()
                if r.priority == priority
            ])

        recent_history = sorted(
            self.review_history,
            key=lambda r: r.reviewed_at or r.created_at,
            reverse=True
        )[:20]

        return {
            "pending_reviews": len(self.pending_reviews),
            "pending_by_priority": pending_by_priority,
            "total_historical": len(self.review_history),
            "recent_reviews": [
                {
                    "id": r.id,
                    "type": r.review_type.value,
                    "title": r.title,
                    "status": r.status.value,
                    "priority": r.priority.value,
                    "created_at": r.created_at.isoformat(),
                    "reviewer": r.reviewer
                }
                for r in recent_history
            ]
        }