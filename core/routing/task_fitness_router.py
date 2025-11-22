"""
Task Fitness Router - Main routing engine with fitness scoring.

Routes tasks to agents based on:
- 0.5 * capability_match (domain expertise)
- 0.3 * (1 - normalized_load) (lower load = higher score)
- 0.2 * historical_win_rate (past success)
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .capability_matrix import AgentCapabilityMatrix, TaskDomain
from .load_tracker import LoadTracker
from .win_rate_tracker import WinRateTracker

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    selected_agent: str
    fitness_score: float
    capability_score: float
    load_score: float
    win_rate_score: float
    all_scores: Dict[str, float]  # agent_name -> fitness_score
    decision_time_ms: float


class TaskFitnessRouter:
    """
    Intelligent task router using fitness scoring.

    Fitness Formula:
        fitness_score = (
            0.5 * capability_match +
            0.3 * (1 - normalized_load) +
            0.2 * historical_win_rate
        )

    Performance Target: <50ms routing decision
    """

    def __init__(
        self,
        capability_matrix: AgentCapabilityMatrix,
        load_tracker: LoadTracker,
        win_rate_tracker: WinRateTracker,
        capability_weight: float = 0.5,
        load_weight: float = 0.3,
        win_rate_weight: float = 0.2
    ):
        """
        Initialize task fitness router.

        Args:
            capability_matrix: Agent capability definitions
            load_tracker: Agent load tracking
            win_rate_tracker: Historical win rate tracking
            capability_weight: Weight for capability matching (default: 0.5)
            load_weight: Weight for load balancing (default: 0.3)
            win_rate_weight: Weight for historical win rate (default: 0.2)
        """
        self.capability_matrix = capability_matrix
        self.load_tracker = load_tracker
        self.win_rate_tracker = win_rate_tracker

        # Routing weights
        self.capability_weight = capability_weight
        self.load_weight = load_weight
        self.win_rate_weight = win_rate_weight

        # Validate weights sum to 1.0
        total_weight = capability_weight + load_weight + win_rate_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Routing weights sum to {total_weight:.2f}, not 1.0. "
                f"Normalizing weights."
            )
            self.capability_weight /= total_weight
            self.load_weight /= total_weight
            self.win_rate_weight /= total_weight

        logger.info(
            f"TaskFitnessRouter initialized: "
            f"capability={self.capability_weight:.2f}, "
            f"load={self.load_weight:.2f}, "
            f"win_rate={self.win_rate_weight:.2f}"
        )

    async def route_task(
        self,
        task_domain: TaskDomain,
        candidate_agents: Optional[List[str]] = None
    ) -> RoutingDecision:
        """
        Route task to best agent using fitness scoring.

        Args:
            task_domain: Task domain for capability matching
            candidate_agents: Optional list of agents to consider (uses all if None)

        Returns:
            RoutingDecision with selected agent and scores

        Performance: <50ms
        """
        start_time = time.time()

        # Get candidate agents
        if candidate_agents is None:
            candidate_agents = list(self.capability_matrix.capabilities.keys())

        if not candidate_agents:
            raise ValueError("No candidate agents available")

        # Calculate fitness scores for all candidates
        fitness_scores: Dict[str, Dict[str, float]] = {}

        for agent_name in candidate_agents:
            # 1. Capability score (domain expertise 0.0-1.0)
            capability_score = self.capability_matrix.get_expertise_score(
                agent_name,
                task_domain
            )

            # 2. Load score (1 - normalized_load, so lower load = higher score)
            normalized_load = await self.load_tracker.get_normalized_load(agent_name)
            load_score = 1.0 - normalized_load

            # 3. Win rate score (historical success rate 0.0-1.0)
            win_rate_score = await self.win_rate_tracker.get_win_rate(agent_name)

            # Calculate weighted fitness score
            fitness_score = (
                self.capability_weight * capability_score +
                self.load_weight * load_score +
                self.win_rate_weight * win_rate_score
            )

            fitness_scores[agent_name] = {
                "fitness": fitness_score,
                "capability": capability_score,
                "load": load_score,
                "win_rate": win_rate_score
            }

            logger.debug(
                f"{agent_name}: fitness={fitness_score:.3f} "
                f"(cap={capability_score:.2f}, "
                f"load={load_score:.2f}, "
                f"win={win_rate_score:.2f})"
            )

        # Select agent with highest fitness score
        best_agent = max(
            fitness_scores.keys(),
            key=lambda agent: fitness_scores[agent]["fitness"]
        )

        best_scores = fitness_scores[best_agent]

        # Calculate decision time
        decision_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Routed {task_domain.value} → {best_agent} "
            f"(fitness={best_scores['fitness']:.3f}, "
            f"time={decision_time_ms:.1f}ms)"
        )

        return RoutingDecision(
            selected_agent=best_agent,
            fitness_score=best_scores["fitness"],
            capability_score=best_scores["capability"],
            load_score=best_scores["load"],
            win_rate_score=best_scores["win_rate"],
            all_scores={
                agent: scores["fitness"]
                for agent, scores in fitness_scores.items()
            },
            decision_time_ms=decision_time_ms
        )

    async def route_task_with_fallback(
        self,
        task_domain: TaskDomain,
        primary_agents: List[str],
        fallback_agents: Optional[List[str]] = None
    ) -> RoutingDecision:
        """
        Route task with fallback logic.

        Tries primary agents first, falls back if all unavailable.

        Args:
            task_domain: Task domain
            primary_agents: Primary agent candidates
            fallback_agents: Fallback agents if primaries unavailable

        Returns:
            RoutingDecision
        """
        # Try primary agents
        decision = await self.route_task(task_domain, primary_agents)

        # Check if selected agent is at capacity
        load = await self.load_tracker.get_load(decision.selected_agent)

        if load.is_at_capacity:
            logger.warning(
                f"{decision.selected_agent} is at capacity. "
                f"Trying fallback agents."
            )

            if fallback_agents:
                decision = await self.route_task(task_domain, fallback_agents)
            else:
                logger.warning("No fallback agents configured. Using best-effort.")

        return decision

    def get_routing_weights(self) -> Dict[str, float]:
        """Get current routing weights."""
        return {
            "capability": self.capability_weight,
            "load": self.load_weight,
            "win_rate": self.win_rate_weight
        }

    def update_routing_weights(
        self,
        capability_weight: Optional[float] = None,
        load_weight: Optional[float] = None,
        win_rate_weight: Optional[float] = None
    ) -> None:
        """
        Update routing weights.

        Weights will be normalized to sum to 1.0.

        Args:
            capability_weight: New capability weight
            load_weight: New load weight
            win_rate_weight: New win rate weight
        """
        if capability_weight is not None:
            self.capability_weight = capability_weight

        if load_weight is not None:
            self.load_weight = load_weight

        if win_rate_weight is not None:
            self.win_rate_weight = win_rate_weight

        # Normalize weights
        total_weight = (
            self.capability_weight +
            self.load_weight +
            self.win_rate_weight
        )

        self.capability_weight /= total_weight
        self.load_weight /= total_weight
        self.win_rate_weight /= total_weight

        logger.info(
            f"Updated routing weights: "
            f"capability={self.capability_weight:.2f}, "
            f"load={self.load_weight:.2f}, "
            f"win_rate={self.win_rate_weight:.2f}"
        )
