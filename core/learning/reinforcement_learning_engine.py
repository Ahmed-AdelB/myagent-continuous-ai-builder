"""
Reinforcement Learning Engine - GPT-5 Recommendation #6
RL Feedback Loops for Adaptive Optimization

Enhances the learning engine with reinforcement learning signals where
successful iterations yield positive rewards to bias future agent strategies.
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import pickle
from pathlib import Path
from collections import defaultdict, deque

from core.evaluation.iteration_quality_framework import IterationQualityScore

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions that agents can take"""
    CODE_REFACTOR = "code_refactor"
    TEST_GENERATION = "test_generation"
    BUG_FIX = "bug_fix"
    FEATURE_ADDITION = "feature_addition"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DOCUMENTATION_UPDATE = "documentation_update"
    ARCHITECTURE_CHANGE = "architecture_change"
    DEPENDENCY_UPDATE = "dependency_update"


class RewardType(Enum):
    """Types of rewards/penalties"""
    QUALITY_IMPROVEMENT = "quality_improvement"
    PERFORMANCE_GAIN = "performance_gain"
    TEST_COVERAGE_INCREASE = "test_coverage_increase"
    BUG_REDUCTION = "bug_reduction"
    USER_SATISFACTION = "user_satisfaction"
    EFFICIENCY_GAIN = "efficiency_gain"
    MAINTAINABILITY_IMPROVEMENT = "maintainability_improvement"


@dataclass
class Action:
    """Action taken by an agent"""
    id: str
    agent_name: str
    action_type: ActionType
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: datetime
    iteration_id: int


@dataclass
class Reward:
    """Reward signal for an action"""
    action_id: str
    reward_type: RewardType
    value: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    feedback: str
    timestamp: datetime


@dataclass
class StateRepresentation:
    """Representation of the system state"""
    iteration_id: int
    quality_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    error_count: int
    test_coverage: float
    code_complexity: float
    recent_actions: List[str]
    timestamp: datetime


@dataclass
class PolicyUpdate:
    """Update to an agent's policy"""
    agent_name: str
    action_type: ActionType
    old_probability: float
    new_probability: float
    learning_rate: float
    update_reason: str
    timestamp: datetime


class AgentPolicy:
    """Policy for an individual agent"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.action_probabilities: Dict[ActionType, float] = {}
        self.action_preferences: Dict[str, float] = {}  # Contextual preferences
        self.success_history: Dict[ActionType, deque] = {}
        self.learning_rate = 0.1

        # Initialize with uniform probabilities
        for action_type in ActionType:
            self.action_probabilities[action_type] = 1.0 / len(ActionType)
            self.success_history[action_type] = deque(maxlen=100)

    def select_action(
        self,
        available_actions: List[ActionType],
        context: Dict[str, Any] = None
    ) -> ActionType:
        """Select an action based on current policy"""
        if not available_actions:
            return ActionType.CODE_REFACTOR  # Default

        # Get probabilities for available actions
        probs = [self.action_probabilities[action] for action in available_actions]

        # Add contextual bias
        if context:
            for i, action in enumerate(available_actions):
                context_key = f"{action.value}_{context.get('primary_goal', 'general')}"
                bias = self.action_preferences.get(context_key, 0.0)
                probs[i] *= (1.0 + bias)

        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(available_actions)] * len(available_actions)

        # Select action using epsilon-greedy with exploration
        epsilon = 0.1  # 10% exploration
        if np.random.random() < epsilon:
            return np.random.choice(available_actions)
        else:
            return np.random.choice(available_actions, p=probs)

    def update_policy(self, action_type: ActionType, reward: float, context: Dict[str, Any] = None):
        """Update policy based on reward signal"""
        # Update action probability
        old_prob = self.action_probabilities[action_type]

        # Q-learning style update
        current_q = self.action_probabilities[action_type]
        new_q = current_q + self.learning_rate * (reward - current_q)

        self.action_probabilities[action_type] = max(0.01, min(0.99, new_q))

        # Update contextual preferences
        if context:
            context_key = f"{action_type.value}_{context.get('primary_goal', 'general')}"
            old_pref = self.action_preferences.get(context_key, 0.0)
            self.action_preferences[context_key] = old_pref + self.learning_rate * reward * 0.5

        # Record success history
        self.success_history[action_type].append(reward > 0)

        # Normalize probabilities to ensure they sum to reasonable values
        total = sum(self.action_probabilities.values())
        if total > len(ActionType) * 0.8:  # If too high, normalize
            for action in ActionType:
                self.action_probabilities[action] /= (total / len(ActionType))

        logger.debug(f"Updated policy for {self.agent_name}: {action_type.value} {old_prob:.3f} -> {self.action_probabilities[action_type]:.3f}")

    def get_success_rate(self, action_type: ActionType) -> float:
        """Get success rate for an action type"""
        history = self.success_history[action_type]
        if not history:
            return 0.5  # Default neutral rate

        return sum(history) / len(history)


class ReinforcementLearningEngine:
    """
    Reinforcement Learning Engine for Adaptive Agent Optimization

    Implements GPT-5 recommendation for adding RL feedback loops
    to enable adaptive, self-optimizing behavior over time.
    """

    def __init__(self, project_name: str, learning_rate: float = 0.1):
        self.project_name = project_name
        self.learning_rate = learning_rate

        # Agent policies
        self.agent_policies: Dict[str, AgentPolicy] = {}

        # Learning data
        self.action_history: List[Action] = []
        self.reward_history: List[Reward] = []
        self.state_history: List[StateRepresentation] = []
        self.policy_updates: List[PolicyUpdate] = []

        # Reward calculation
        self.reward_functions: Dict[RewardType, Callable] = {}
        self._setup_reward_functions()

        # Persistence
        self.storage_path = Path(f"persistence/rl_engine_{project_name}")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Reinforcement Learning Engine initialized for: {project_name}")

    async def start(self):
        """Start the RL engine and load persisted data"""
        await self._load_persisted_data()
        asyncio.create_task(self._periodic_policy_optimization())
        logger.info("Reinforcement Learning Engine started")

    def record_action(
        self,
        agent_name: str,
        action_type: ActionType,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
        iteration_id: int
    ) -> str:
        """Record an action taken by an agent"""
        action_id = f"action_{len(self.action_history)}"

        action = Action(
            id=action_id,
            agent_name=agent_name,
            action_type=action_type,
            parameters=parameters,
            context=context,
            timestamp=datetime.now(),
            iteration_id=iteration_id
        )

        self.action_history.append(action)

        # Ensure agent has a policy
        if agent_name not in self.agent_policies:
            self.agent_policies[agent_name] = AgentPolicy(agent_name)

        logger.debug(f"Recorded action {action_id}: {agent_name} -> {action_type.value}")
        return action_id

    def record_state(
        self,
        iteration_id: int,
        quality_metrics: Dict[str, float],
        performance_metrics: Dict[str, float],
        error_count: int,
        test_coverage: float,
        code_complexity: float
    ):
        """Record the current system state"""
        # Get recent actions for state representation
        recent_actions = [
            a.id for a in self.action_history[-10:]  # Last 10 actions
            if a.iteration_id >= iteration_id - 2  # Within 2 iterations
        ]

        state = StateRepresentation(
            iteration_id=iteration_id,
            quality_metrics=quality_metrics,
            performance_metrics=performance_metrics,
            error_count=error_count,
            test_coverage=test_coverage,
            code_complexity=code_complexity,
            recent_actions=recent_actions,
            timestamp=datetime.now()
        )

        self.state_history.append(state)
        logger.debug(f"Recorded state for iteration {iteration_id}")

    async def process_iteration_results(self, iteration_quality_score: IterationQualityScore):
        """Process iteration results and generate reward signals"""
        current_iteration = iteration_quality_score.iteration_id

        # Get actions from this iteration
        iteration_actions = [
            a for a in self.action_history
            if a.iteration_id == current_iteration
        ]

        if not iteration_actions:
            logger.warning(f"No actions found for iteration {current_iteration}")
            return

        # Calculate rewards for each action
        for action in iteration_actions:
            rewards = await self._calculate_action_rewards(action, iteration_quality_score)

            for reward in rewards:
                self.reward_history.append(reward)

                # Update agent policy
                agent_policy = self.agent_policies[action.agent_name]
                agent_policy.update_policy(
                    action.action_type,
                    reward.value,
                    action.context
                )

                # Record policy update
                policy_update = PolicyUpdate(
                    agent_name=action.agent_name,
                    action_type=action.action_type,
                    old_probability=agent_policy.action_probabilities[action.action_type],
                    new_probability=agent_policy.action_probabilities[action.action_type],
                    learning_rate=self.learning_rate,
                    update_reason=f"Reward: {reward.value:.3f} for {reward.reward_type.value}",
                    timestamp=datetime.now()
                )

                self.policy_updates.append(policy_update)

        logger.info(f"Processed {len(iteration_actions)} actions from iteration {current_iteration}")

    async def get_action_recommendation(
        self,
        agent_name: str,
        available_actions: List[ActionType],
        context: Dict[str, Any] = None
    ) -> Tuple[ActionType, float]:
        """Get action recommendation for an agent"""
        if agent_name not in self.agent_policies:
            self.agent_policies[agent_name] = AgentPolicy(agent_name)

        policy = self.agent_policies[agent_name]
        recommended_action = policy.select_action(available_actions, context)

        # Calculate confidence based on success history
        confidence = policy.get_success_rate(recommended_action)

        logger.debug(f"Recommended action for {agent_name}: {recommended_action.value} (confidence: {confidence:.3f})")

        return recommended_action, confidence

    async def _calculate_action_rewards(
        self,
        action: Action,
        iteration_quality_score: IterationQualityScore
    ) -> List[Reward]:
        """Calculate reward signals for an action based on iteration results"""
        rewards = []

        # Quality improvement reward
        if iteration_quality_score.improvement_delta > 0:
            reward_value = min(1.0, iteration_quality_score.improvement_delta * 5)  # Scale improvement
            rewards.append(Reward(
                action_id=action.id,
                reward_type=RewardType.QUALITY_IMPROVEMENT,
                value=reward_value,
                confidence=0.8,
                feedback=f"Quality improved by {iteration_quality_score.improvement_delta:.3f}",
                timestamp=datetime.now()
            ))

        # Test coverage reward
        test_metrics = [m for m in iteration_quality_score.metrics if "coverage" in m.name]
        if test_metrics:
            avg_coverage = sum(m.value for m in test_metrics) / len(test_metrics)
            if avg_coverage > 0.9:  # High coverage
                rewards.append(Reward(
                    action_id=action.id,
                    reward_type=RewardType.TEST_COVERAGE_INCREASE,
                    value=0.8,
                    confidence=0.9,
                    feedback=f"Excellent test coverage: {avg_coverage:.2f}",
                    timestamp=datetime.now()
                ))

        # Performance reward
        perf_metrics = [m for m in iteration_quality_score.metrics if m.metric_type.value == "performance"]
        if perf_metrics:
            avg_perf = sum(m.value for m in perf_metrics) / len(perf_metrics)
            if avg_perf > 0.85:  # Good performance
                rewards.append(Reward(
                    action_id=action.id,
                    reward_type=RewardType.PERFORMANCE_GAIN,
                    value=0.7,
                    confidence=0.8,
                    feedback=f"Good performance metrics: {avg_perf:.2f}",
                    timestamp=datetime.now()
                ))

        # Bug reduction reward (if this was a bug fix action)
        if action.action_type == ActionType.BUG_FIX:
            # Check if error count decreased
            current_state = self.state_history[-1] if self.state_history else None
            previous_states = [s for s in self.state_history if s.iteration_id < action.iteration_id]

            if current_state and previous_states:
                prev_state = previous_states[-1]
                if current_state.error_count < prev_state.error_count:
                    reduction_rate = (prev_state.error_count - current_state.error_count) / max(1, prev_state.error_count)
                    rewards.append(Reward(
                        action_id=action.id,
                        reward_type=RewardType.BUG_REDUCTION,
                        value=min(1.0, reduction_rate * 2),
                        confidence=0.9,
                        feedback=f"Reduced errors by {reduction_rate:.1%}",
                        timestamp=datetime.now()
                    ))

        # Negative rewards for poor outcomes
        if iteration_quality_score.improvement_delta < -0.1:  # Significant regression
            rewards.append(Reward(
                action_id=action.id,
                reward_type=RewardType.QUALITY_IMPROVEMENT,
                value=-0.8,
                confidence=0.7,
                feedback=f"Quality regression: {iteration_quality_score.improvement_delta:.3f}",
                timestamp=datetime.now()
            ))

        return rewards

    def _setup_reward_functions(self):
        """Setup reward calculation functions"""
        # This could be expanded with custom reward functions
        pass

    async def _periodic_policy_optimization(self):
        """Periodically optimize agent policies"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Analyze policy performance
                for agent_name, policy in self.agent_policies.items():
                    await self._analyze_policy_performance(agent_name, policy)

                # Save state
                await self._persist_data()

            except Exception as e:
                logger.error(f"Error in periodic policy optimization: {e}")

    async def _analyze_policy_performance(self, agent_name: str, policy: AgentPolicy):
        """Analyze and optimize an agent's policy performance"""
        # Get recent actions for this agent
        recent_actions = [
            a for a in self.action_history[-200:]  # Last 200 actions
            if a.agent_name == agent_name
        ]

        if len(recent_actions) < 10:  # Not enough data
            return

        # Analyze success rates by action type
        action_performance = defaultdict(list)
        for action in recent_actions:
            # Find rewards for this action
            action_rewards = [r for r in self.reward_history if r.action_id == action.id]
            if action_rewards:
                avg_reward = sum(r.value for r in action_rewards) / len(action_rewards)
                action_performance[action.action_type].append(avg_reward)

        # Update policy based on performance
        for action_type, rewards in action_performance.items():
            if len(rewards) >= 5:  # Enough samples
                avg_performance = sum(rewards) / len(rewards)

                # Adjust learning rate based on performance variance
                variance = np.var(rewards)
                adaptive_lr = self.learning_rate * (1.0 if variance < 0.1 else 0.5)

                # Update policy
                policy.learning_rate = adaptive_lr

        logger.debug(f"Analyzed policy performance for {agent_name}")

    async def _persist_data(self):
        """Persist learning data to disk"""
        try:
            # Save policies
            policies_file = self.storage_path / "agent_policies.pkl"
            with open(policies_file, 'wb') as f:
                pickle.dump(self.agent_policies, f)

            # Save learning history (recent only to prevent excessive growth)
            history_data = {
                "action_history": self.action_history[-1000:],  # Last 1000 actions
                "reward_history": self.reward_history[-1000:],  # Last 1000 rewards
                "state_history": self.state_history[-100:],     # Last 100 states
                "policy_updates": self.policy_updates[-500:]    # Last 500 updates
            }

            history_file = self.storage_path / "learning_history.json"
            with open(history_file, 'w') as f:
                # Custom serialization for datetime and enums
                def serialize_obj(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, (ActionType, RewardType)):
                        return obj.value
                    elif hasattr(obj, '__dict__'):
                        return asdict(obj)
                    else:
                        return str(obj)

                json.dump(history_data, f, default=serialize_obj, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist RL data: {e}")

    async def _load_persisted_data(self):
        """Load persisted learning data"""
        try:
            # Load policies
            policies_file = self.storage_path / "agent_policies.pkl"
            if policies_file.exists():
                with open(policies_file, 'rb') as f:
                    self.agent_policies = pickle.load(f)

            # Load learning history
            history_file = self.storage_path / "learning_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)

                # Reconstruct objects (simplified for demo)
                # In production, would need full reconstruction logic
                logger.info(f"Loaded learning history with {len(history_data.get('action_history', []))} actions")

        except Exception as e:
            logger.error(f"Failed to load persisted RL data: {e}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = {
            "total_actions": len(self.action_history),
            "total_rewards": len(self.reward_history),
            "active_agents": len(self.agent_policies),
            "policy_updates": len(self.policy_updates)
        }

        # Agent performance
        agent_stats = {}
        for agent_name, policy in self.agent_policies.items():
            agent_actions = [a for a in self.action_history if a.agent_name == agent_name]
            agent_rewards = [
                r for r in self.reward_history
                if any(a.id == r.action_id and a.agent_name == agent_name for a in agent_actions)
            ]

            avg_reward = sum(r.value for r in agent_rewards) / len(agent_rewards) if agent_rewards else 0.0

            agent_stats[agent_name] = {
                "total_actions": len(agent_actions),
                "average_reward": avg_reward,
                "success_rates": {
                    action_type.value: policy.get_success_rate(action_type)
                    for action_type in ActionType
                }
            }

        stats["agent_performance"] = agent_stats

        return stats