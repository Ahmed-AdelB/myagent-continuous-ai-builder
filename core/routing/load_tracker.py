"""
Load Tracker - Real-time agent workload tracking.

Tracks current tasks per agent for load balancing.
"""

from dataclasses import dataclass
from typing import Dict
from datetime import datetime
import asyncio


@dataclass
class AgentLoad:
    """Agent load information."""
    agent_name: str
    active_tasks: int
    max_capacity: int
    last_updated: datetime

    @property
    def normalized_load(self) -> float:
        """Get normalized load 0.0-1.0 (1.0 = fully loaded)."""
        if self.max_capacity == 0:
            return 0.0
        return min(1.0, self.active_tasks / self.max_capacity)

    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return max(0, self.max_capacity - self.active_tasks)

    @property
    def is_at_capacity(self) -> bool:
        """Check if agent is at max capacity."""
        return self.active_tasks >= self.max_capacity


class LoadTracker:
    """
    Tracks real-time agent workload.

    Features:
    - Track active tasks per agent
    - Configurable max capacity per agent
    - Normalized load for routing decisions
    - Thread-safe updates
    """

    def __init__(self, default_max_capacity: int = 5):
        """
        Initialize load tracker.

        Args:
            default_max_capacity: Default max concurrent tasks per agent
        """
        self.default_max_capacity = default_max_capacity
        self.loads: Dict[str, AgentLoad] = {}
        self._lock = asyncio.Lock()

    async def initialize_agent(
        self,
        agent_name: str,
        max_capacity: int = None
    ) -> None:
        """
        Initialize agent load tracking.

        Args:
            agent_name: Agent identifier
            max_capacity: Max concurrent tasks (uses default if None)
        """
        async with self._lock:
            if agent_name not in self.loads:
                self.loads[agent_name] = AgentLoad(
                    agent_name=agent_name,
                    active_tasks=0,
                    max_capacity=max_capacity or self.default_max_capacity,
                    last_updated=datetime.now()
                )

    async def increment_load(self, agent_name: str) -> None:
        """
        Increment active task count for agent.

        Args:
            agent_name: Agent identifier
        """
        async with self._lock:
            if agent_name not in self.loads:
                await self.initialize_agent(agent_name)

            self.loads[agent_name].active_tasks += 1
            self.loads[agent_name].last_updated = datetime.now()

    async def decrement_load(self, agent_name: str) -> None:
        """
        Decrement active task count for agent.

        Args:
            agent_name: Agent identifier
        """
        async with self._lock:
            if agent_name not in self.loads:
                return

            self.loads[agent_name].active_tasks = max(
                0,
                self.loads[agent_name].active_tasks - 1
            )
            self.loads[agent_name].last_updated = datetime.now()

    async def get_load(self, agent_name: str) -> AgentLoad:
        """
        Get load information for agent.

        Args:
            agent_name: Agent identifier

        Returns:
            AgentLoad object
        """
        async with self._lock:
            if agent_name not in self.loads:
                await self.initialize_agent(agent_name)
            return self.loads[agent_name]

    async def get_normalized_load(self, agent_name: str) -> float:
        """
        Get normalized load for agent (0.0-1.0).

        Args:
            agent_name: Agent identifier

        Returns:
            Normalized load (1.0 = fully loaded)
        """
        load = await self.get_load(agent_name)
        return load.normalized_load

    async def get_all_loads(self) -> Dict[str, AgentLoad]:
        """Get load information for all agents."""
        async with self._lock:
            return dict(self.loads)

    async def get_least_loaded_agent(self, agent_names: list) -> str:
        """
        Get least loaded agent from list.

        Args:
            agent_names: List of agent names to consider

        Returns:
            Agent name with lowest load
        """
        loads = []
        for agent_name in agent_names:
            load = await self.get_load(agent_name)
            loads.append((agent_name, load.normalized_load))

        loads.sort(key=lambda x: x[1])
        return loads[0][0] if loads else agent_names[0]

    async def reset_load(self, agent_name: str) -> None:
        """Reset load for agent to zero."""
        async with self._lock:
            if agent_name in self.loads:
                self.loads[agent_name].active_tasks = 0
                self.loads[agent_name].last_updated = datetime.now()

    async def set_max_capacity(self, agent_name: str, max_capacity: int) -> None:
        """Update max capacity for agent."""
        async with self._lock:
            if agent_name not in self.loads:
                await self.initialize_agent(agent_name, max_capacity)
            else:
                self.loads[agent_name].max_capacity = max_capacity
                self.loads[agent_name].last_updated = datetime.now()
