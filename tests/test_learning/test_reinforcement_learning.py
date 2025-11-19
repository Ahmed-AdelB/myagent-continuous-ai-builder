"""
Tests for Reinforcement Learning Engine

Validates the RL engine's ability to:
- Learn from agent actions and outcomes
- Optimize agent selection over time
- Balance exploration vs exploitation
"""

import pytest
import asyncio
from datetime import datetime
from core.learning.reinforcement_learning_engine import RLEngine, State, Action, Reward


class TestRLEngine:
    """Test suite for Reinforcement Learning Engine"""

    @pytest.fixture
    def rl_engine(self):
        """Create fresh RL engine instance"""
        return RLEngine(
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.2
        )

    @pytest.fixture
    def sample_state(self):
        """Sample state for testing"""
        return State(
            task_type='implement_feature',
            complexity=5,
            dependencies=2,
            current_metrics={'test_coverage': 75}
        )

    @pytest.fixture
    def sample_actions(self):
        """Sample actions (agent selections)"""
        return [
            Action(agent='coder', confidence=0.8),
            Action(agent='tester', confidence=0.9),
            Action(agent='debugger', confidence=0.7)
        ]

    def test_initialization(self, rl_engine):
        """Test RL engine initializes with correct parameters"""
        assert rl_engine.learning_rate == 0.1
        assert rl_engine.discount_factor == 0.9
        assert rl_engine.exploration_rate == 0.2
        assert hasattr(rl_engine, 'q_table')

    @pytest.mark.asyncio
    async def test_action_selection_exploration(self, rl_engine, sample_state):
        """Test action selection with exploration"""
        rl_engine.exploration_rate = 1.0  # Always explore

        actions = []
        for _ in range(10):
            action = await rl_engine.select_action(sample_state)
            actions.append(action)

        # Should have variety due to exploration
        unique_actions = len(set(a.agent for a in actions))
        assert unique_actions > 1

    @pytest.mark.asyncio
    async def test_action_selection_exploitation(self, rl_engine, sample_state):
        """Test action selection with exploitation"""
        rl_engine.exploration_rate = 0.0  # Never explore

        # Train with high rewards for 'coder' agent
        coder_action = Action(agent='coder', confidence=0.9)
        for _ in range(10):
            await rl_engine.update_q_value(
                state=sample_state,
                action=coder_action,
                reward=Reward(value=1.0, success=True),
                next_state=sample_state
            )

        # Should always select 'coder' now
        selected = await rl_engine.select_action(sample_state)
        assert selected.agent == 'coder'

    @pytest.mark.asyncio
    async def test_q_value_update(self, rl_engine, sample_state):
        """Test Q-value updates with rewards"""
        action = Action(agent='tester', confidence=0.8)

        # Initial Q-value
        initial_q = rl_engine.get_q_value(sample_state, action)

        # Positive reward
        await rl_engine.update_q_value(
            state=sample_state,
            action=action,
            reward=Reward(value=1.0, success=True),
            next_state=sample_state
        )

        # Q-value should increase
        updated_q = rl_engine.get_q_value(sample_state, action)
        assert updated_q > initial_q

    @pytest.mark.asyncio
    async def test_learning_from_failure(self, rl_engine, sample_state):
        """Test that engine learns from failures"""
        bad_action = Action(agent='bad_agent', confidence=0.5)

        # Initial Q-value
        initial_q = rl_engine.get_q_value(sample_state, bad_action)

        # Negative rewards
        for _ in range(5):
            await rl_engine.update_q_value(
                state=sample_state,
                action=bad_action,
                reward=Reward(value=-0.5, success=False),
                next_state=sample_state
            )

        # Q-value should decrease
        updated_q = rl_engine.get_q_value(sample_state, bad_action)
        assert updated_q < initial_q

    @pytest.mark.asyncio
    async def test_exploration_decay(self, rl_engine):
        """Test exploration rate decays over time"""
        initial_exploration = rl_engine.exploration_rate

        # Simulate many episodes
        for _ in range(100):
            rl_engine.decay_exploration()

        final_exploration = rl_engine.exploration_rate
        assert final_exploration < initial_exploration
        assert final_exploration >= rl_engine.min_exploration_rate

    def test_state_serialization(self, sample_state):
        """Test state can be serialized for Q-table keys"""
        state_key = sample_state.to_key()
        assert isinstance(state_key, str)
        assert 'implement_feature' in state_key

    @pytest.mark.asyncio
    async def test_best_action_selection(self, rl_engine, sample_state):
        """Test selecting best action based on Q-values"""
        # Train different actions
        actions = [
            Action(agent='agent1', confidence=0.8),
            Action(agent='agent2', confidence=0.9),
            Action(agent='agent3', confidence=0.7)
        ]

        # Give agent2 highest rewards
        for _ in range(10):
            await rl_engine.update_q_value(
                sample_state,
                actions[1],
                Reward(value=1.0, success=True),
                sample_state
            )

        # Get best action
        best = rl_engine.get_best_action(sample_state, actions)
        assert best.agent == 'agent2'

    @pytest.mark.asyncio
    async def test_convergence(self, rl_engine):
        """Test Q-values converge to stable values"""
        state = State(task_type='test', complexity=1, dependencies=0, current_metrics={})
        action = Action(agent='test_agent', confidence=1.0)

        q_values = []
        for _ in range(100):
            await rl_engine.update_q_value(
                state,
                action,
                Reward(value=1.0, success=True),
                state
            )
            q_values.append(rl_engine.get_q_value(state, action))

        # Check convergence (last 10 values should be similar)
        recent_values = q_values[-10:]
        variance = sum((x - sum(recent_values) / len(recent_values)) ** 2 for x in recent_values) / len(recent_values)
        assert variance < 0.01  # Low variance indicates convergence

    def test_save_and_load(self, rl_engine, tmp_path):
        """Test saving and loading Q-table"""
        state = State(task_type='save_test', complexity=1, dependencies=0, current_metrics={})
        action = Action(agent='test', confidence=0.9)

        # Train
        for _ in range(5):
            rl_engine.update_q_value_sync(
                state,
                action,
                Reward(value=1.0, success=True),
                state
            )

        # Save
        save_path = tmp_path / "q_table.json"
        rl_engine.save(str(save_path))

        # Load into new engine
        new_engine = RLEngine()
        new_engine.load(str(save_path))

        # Should have same Q-value
        original_q = rl_engine.get_q_value(state, action)
        loaded_q = new_engine.get_q_value(state, action)
        assert abs(original_q - loaded_q) < 0.001
