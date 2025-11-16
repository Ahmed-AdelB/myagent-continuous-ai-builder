"""
Meta-Governor Agent - GPT-5 Recommendation #1
Safe Continuous Operation and Iteration Control

Prevents runaway iteration, resource exhaustion, and feedback loops.
Monitors iteration health, resource use, and convergence metrics.
"""

import asyncio
import time
import psutil
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

from core.memory.project_ledger import ProjectLedger
from core.monitoring.system_observability import SystemObservabilityMonitor

logger = logging.getLogger(__name__)


@dataclass
class IterationHealth:
    """Health metrics for a single iteration"""
    iteration_id: int
    duration_seconds: float
    resource_usage: Dict[str, float]
    quality_score: float
    convergence_score: float
    error_count: int
    warning_count: int
    timestamp: datetime


@dataclass
class GovernanceThresholds:
    """Configurable thresholds for governance decisions"""
    max_iteration_duration: int = 3600  # 1 hour max per iteration
    max_memory_usage_mb: int = 8192  # 8GB max memory
    max_cpu_usage_percent: float = 90.0  # 90% max CPU
    min_quality_improvement: float = 0.01  # 1% minimum improvement
    convergence_plateau_threshold: int = 5  # iterations without improvement
    error_spike_threshold: int = 10  # max errors per iteration
    resource_exhaustion_threshold: float = 0.95  # 95% resource usage


class MetaGovernorAgent:
    """
    Meta-Governor Agent for Safe Continuous Operation

    Monitors and controls the continuous iteration process to prevent:
    - Runaway iteration cycles
    - Resource exhaustion
    - Quality regression
    - System instability
    """

    def __init__(self, project_name: str, thresholds: Optional[GovernanceThresholds] = None):
        self.project_name = project_name
        self.thresholds = thresholds or GovernanceThresholds()

        # Initialize monitoring systems
        self.project_ledger = ProjectLedger(project_name)
        self.observability = SystemObservabilityMonitor()

        # State tracking
        self.current_iteration = 0
        self.iteration_history: List[IterationHealth] = []
        self.governance_active = True
        self.emergency_stop = False

        # Convergence tracking
        self.quality_history: List[float] = []
        self.convergence_plateau_count = 0

        logger.info(f"Meta-Governor initialized for project: {project_name}")

    async def start_iteration_monitoring(self, iteration_id: int) -> bool:
        """
        Start monitoring a new iteration

        Returns:
            bool: True if iteration should proceed, False if blocked
        """
        self.current_iteration = iteration_id

        # Pre-iteration safety checks
        if not await self._pre_iteration_safety_check():
            logger.warning(f"Iteration {iteration_id} blocked by pre-iteration safety checks")
            return False

        # Record iteration start
        self.iteration_start_time = time.time()
        logger.info(f"Iteration {iteration_id} monitoring started")

        return True

    async def monitor_iteration_progress(self) -> Dict[str, any]:
        """
        Continuously monitor iteration progress

        Returns:
            Dict with monitoring status and recommendations
        """
        if not self.governance_active:
            return {"status": "inactive"}

        # Collect current metrics
        resource_usage = self._get_resource_usage()
        duration = time.time() - self.iteration_start_time

        # Check safety conditions
        safety_status = await self._check_safety_conditions(duration, resource_usage)

        # Check for emergency conditions
        if safety_status.get("emergency_stop", False):
            self.emergency_stop = True
            await self._trigger_emergency_stop(safety_status.get("reason", "Unknown"))

        return {
            "status": "active",
            "iteration": self.current_iteration,
            "duration": duration,
            "resource_usage": resource_usage,
            "safety_status": safety_status,
            "emergency_stop": self.emergency_stop
        }

    async def complete_iteration(self, quality_score: float, error_count: int, warning_count: int) -> Dict[str, any]:
        """
        Complete iteration monitoring and make governance decisions

        Args:
            quality_score: Overall quality score for this iteration (0.0-1.0)
            error_count: Number of errors encountered
            warning_count: Number of warnings encountered

        Returns:
            Dict with completion status and next action recommendations
        """
        duration = time.time() - self.iteration_start_time
        resource_usage = self._get_resource_usage()

        # Calculate convergence score
        convergence_score = self._calculate_convergence_score(quality_score)

        # Record iteration health
        health = IterationHealth(
            iteration_id=self.current_iteration,
            duration_seconds=duration,
            resource_usage=resource_usage,
            quality_score=quality_score,
            convergence_score=convergence_score,
            error_count=error_count,
            warning_count=warning_count,
            timestamp=datetime.now()
        )

        self.iteration_history.append(health)
        self.quality_history.append(quality_score)

        # Update convergence tracking
        self._update_convergence_tracking(quality_score)

        # Make governance decision
        governance_decision = await self._make_governance_decision(health)

        # Log iteration completion
        logger.info(f"Iteration {self.current_iteration} completed - {governance_decision['action']}")

        return governance_decision

    async def _pre_iteration_safety_check(self) -> bool:
        """Check if it's safe to start a new iteration"""

        # Check system resources
        resource_usage = self._get_resource_usage()

        if resource_usage.get("memory_percent", 0) > self.thresholds.resource_exhaustion_threshold * 100:
            logger.error("Memory usage too high to start iteration")
            return False

        if resource_usage.get("cpu_percent", 0) > self.thresholds.resource_exhaustion_threshold * 100:
            logger.error("CPU usage too high to start iteration")
            return False

        # Check for emergency stop condition
        if self.emergency_stop:
            logger.error("Emergency stop active - blocking iteration")
            return False

        # Check convergence plateau
        if self.convergence_plateau_count >= self.thresholds.convergence_plateau_threshold:
            logger.warning(f"Convergence plateau detected ({self.convergence_plateau_count} iterations)")
            # Allow iteration but with warning

        return True

    async def _check_safety_conditions(self, duration: float, resource_usage: Dict) -> Dict[str, any]:
        """Check safety conditions during iteration"""

        safety_status = {
            "safe": True,
            "warnings": [],
            "emergency_stop": False,
            "reason": None
        }

        # Check duration
        if duration > self.thresholds.max_iteration_duration:
            safety_status["emergency_stop"] = True
            safety_status["reason"] = f"Iteration duration exceeded {self.thresholds.max_iteration_duration}s"

        # Check memory usage
        memory_mb = resource_usage.get("memory_mb", 0)
        if memory_mb > self.thresholds.max_memory_usage_mb:
            safety_status["emergency_stop"] = True
            safety_status["reason"] = f"Memory usage exceeded {self.thresholds.max_memory_usage_mb}MB"

        # Check CPU usage
        cpu_percent = resource_usage.get("cpu_percent", 0)
        if cpu_percent > self.thresholds.max_cpu_usage_percent:
            safety_status["warnings"].append(f"High CPU usage: {cpu_percent:.1f}%")

        return safety_status

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')

            return {
                "memory_mb": memory.used / 1024 / 1024,
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "disk_percent": disk.percent
            }
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}

    def _calculate_convergence_score(self, current_quality: float) -> float:
        """Calculate convergence score based on quality improvement trend"""
        if len(self.quality_history) < 2:
            return 0.5  # Neutral score for first iterations

        recent_scores = self.quality_history[-5:]  # Last 5 iterations

        if len(recent_scores) < 2:
            return 0.5

        # Calculate trend (improvement rate)
        improvements = [recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))]
        avg_improvement = sum(improvements) / len(improvements)

        # Convert to convergence score (0.0 = poor convergence, 1.0 = excellent convergence)
        convergence_score = max(0.0, min(1.0, 0.5 + (avg_improvement * 10)))

        return convergence_score

    def _update_convergence_tracking(self, quality_score: float):
        """Update convergence plateau tracking"""
        if len(self.quality_history) < 2:
            return

        previous_quality = self.quality_history[-2]
        improvement = quality_score - previous_quality

        if improvement < self.thresholds.min_quality_improvement:
            self.convergence_plateau_count += 1
        else:
            self.convergence_plateau_count = 0  # Reset counter

    async def _make_governance_decision(self, health: IterationHealth) -> Dict[str, any]:
        """Make governance decision based on iteration health"""

        decision = {
            "action": "continue",  # continue, pause, stop, adjust
            "reason": "",
            "recommended_changes": [],
            "next_iteration_delay": 0  # seconds to wait before next iteration
        }

        # Check for convergence plateau
        if self.convergence_plateau_count >= self.thresholds.convergence_plateau_threshold:
            decision["action"] = "pause"
            decision["reason"] = f"Convergence plateau detected ({self.convergence_plateau_count} iterations)"
            decision["next_iteration_delay"] = 300  # 5-minute pause
            decision["recommended_changes"].append("Consider changing iteration strategy")

        # Check for error spike
        if health.error_count > self.thresholds.error_spike_threshold:
            decision["action"] = "pause"
            decision["reason"] = f"Error spike detected ({health.error_count} errors)"
            decision["next_iteration_delay"] = 600  # 10-minute pause
            decision["recommended_changes"].append("Investigate and fix critical errors")

        # Check for quality regression
        if len(self.quality_history) > 1:
            previous_quality = self.quality_history[-2]
            if health.quality_score < previous_quality - 0.1:  # 10% regression
                decision["action"] = "adjust"
                decision["reason"] = f"Quality regression detected"
                decision["recommended_changes"].append("Revert last changes and re-analyze")

        # Check for resource pressure
        memory_percent = health.resource_usage.get("memory_percent", 0)
        if memory_percent > 80:  # 80% memory usage
            decision["next_iteration_delay"] = max(decision["next_iteration_delay"], 60)
            decision["recommended_changes"].append("Optimize memory usage")

        return decision

    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop procedure"""
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")

        # Record emergency stop in project ledger
        await self.project_ledger.record_decision(
            self.current_iteration,
            "meta_governor",
            "emergency_stop",
            f"Emergency stop triggered: {reason}"
        )

        # Stop all agents (implementation depends on agent communication system)
        # This would integrate with the Agent Communication Bus from GPT-5 Priority #3

        # Send alerts
        await self._send_emergency_alert(reason)

    async def _send_emergency_alert(self, reason: str):
        """Send emergency alert to administrators"""
        # Implementation would integrate with monitoring/alerting system
        logger.critical(f"EMERGENCY ALERT: {reason}")

        # Could integrate with:
        # - GCP Cloud Monitoring alerts
        # - Slack/email notifications
        # - PagerDuty incidents

    def get_governance_status(self) -> Dict[str, any]:
        """Get current governance status and health metrics"""
        recent_health = self.iteration_history[-5:] if self.iteration_history else []

        return {
            "governance_active": self.governance_active,
            "emergency_stop": self.emergency_stop,
            "current_iteration": self.current_iteration,
            "convergence_plateau_count": self.convergence_plateau_count,
            "recent_iterations": [asdict(h) for h in recent_health],
            "thresholds": asdict(self.thresholds)
        }

    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop (requires manual intervention)"""
        if self.emergency_stop:
            logger.info("Emergency stop reset by administrator")
            self.emergency_stop = False
            return True
        return False