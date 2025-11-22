"""
Win Rate Tracker - Historical success rate tracking.

Tracks agent success rates from TaskLedger for routing decisions.
"""

from typing import Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import asyncio

from ..resilience.task_ledger import TaskLedger, TaskState


@dataclass
class WinRateStats:
    """Agent win rate statistics."""
    agent_name: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    win_rate: float  # 0.0-1.0

    @property
    def loss_rate(self) -> float:
        """Get loss rate (1.0 - win_rate)."""
        return 1.0 - self.win_rate


class WinRateTracker:
    """
    Tracks historical success rates from TaskLedger.

    Features:
    - Query TaskLedger for task outcomes
    - Calculate win rates per agent
    - Cache win rates with periodic refresh
    - Exponential moving average for recent performance
    """

    def __init__(
        self,
        task_ledger: TaskLedger,
        cache_ttl_seconds: float = 60.0,
        ema_alpha: float = 0.3
    ):
        """
        Initialize win rate tracker.

        Args:
            task_ledger: TaskLedger instance for querying task history
            cache_ttl_seconds: Cache time-to-live in seconds
            ema_alpha: Exponential moving average smoothing factor (0.0-1.0)
        """
        self.task_ledger = task_ledger
        self.cache_ttl_seconds = cache_ttl_seconds
        self.ema_alpha = ema_alpha

        self.win_rates: Dict[str, WinRateStats] = {}
        self.last_refresh: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get_win_rate(self, agent_name: str) -> float:
        """
        Get win rate for agent (0.0-1.0).

        Returns:
            Win rate (successful_tasks / total_tasks)
            Returns 0.5 if no task history
        """
        stats = await self.get_stats(agent_name)
        return stats.win_rate

    async def get_stats(self, agent_name: str) -> WinRateStats:
        """
        Get win rate statistics for agent.

        Args:
            agent_name: Agent identifier

        Returns:
            WinRateStats object
        """
        # Check cache
        import time
        current_time = time.time()

        if agent_name in self.win_rates:
            last_refresh = self.last_refresh.get(agent_name, 0.0)
            if current_time - last_refresh < self.cache_ttl_seconds:
                # Cache hit
                return self.win_rates[agent_name]

        # Cache miss or expired - refresh from TaskLedger
        async with self._lock:
            stats = await self._calculate_win_rate(agent_name)
            self.win_rates[agent_name] = stats
            self.last_refresh[agent_name] = current_time
            return stats

    async def _calculate_win_rate(self, agent_name: str) -> WinRateStats:
        """
        Calculate win rate from TaskLedger.

        Args:
            agent_name: Agent identifier

        Returns:
            WinRateStats object
        """
        # Query TaskLedger for agent tasks
        tasks = await self.task_ledger.get_tasks_by_agent(agent_name)

        if not tasks:
            # No history - return neutral 0.5 win rate
            return WinRateStats(
                agent_name=agent_name,
                total_tasks=0,
                successful_tasks=0,
                failed_tasks=0,
                win_rate=0.5  # Neutral default
            )

        # Count successes and failures
        successful_tasks = sum(1 for task in tasks if task.state == TaskState.DONE)
        failed_tasks = sum(1 for task in tasks if task.state == TaskState.FAILED)
        total_tasks = len(tasks)

        # Calculate win rate
        if total_tasks == 0:
            win_rate = 0.5
        else:
            win_rate = successful_tasks / total_tasks

        return WinRateStats(
            agent_name=agent_name,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            win_rate=win_rate
        )

    async def update_win_rate_ema(
        self,
        agent_name: str,
        task_succeeded: bool
    ) -> None:
        """
        Update win rate using exponential moving average.

        This provides real-time updates without querying full task history.

        Args:
            agent_name: Agent identifier
            task_succeeded: True if task succeeded, False if failed
        """
        async with self._lock:
            # Get current stats
            stats = await self.get_stats(agent_name)

            # Update with EMA
            new_sample = 1.0 if task_succeeded else 0.0
            new_win_rate = (
                self.ema_alpha * new_sample +
                (1 - self.ema_alpha) * stats.win_rate
            )

            # Update stats
            updated_stats = WinRateStats(
                agent_name=agent_name,
                total_tasks=stats.total_tasks + 1,
                successful_tasks=stats.successful_tasks + (1 if task_succeeded else 0),
                failed_tasks=stats.failed_tasks + (0 if task_succeeded else 1),
                win_rate=new_win_rate
            )

            self.win_rates[agent_name] = updated_stats

            import time
            self.last_refresh[agent_name] = time.time()

    async def refresh_all(self) -> None:
        """Refresh win rates for all tracked agents."""
        async with self._lock:
            agent_names = list(self.win_rates.keys())

            for agent_name in agent_names:
                stats = await self._calculate_win_rate(agent_name)
                self.win_rates[agent_name] = stats

                import time
                self.last_refresh[agent_name] = time.time()

    async def get_all_stats(self) -> Dict[str, WinRateStats]:
        """Get win rate statistics for all agents."""
        async with self._lock:
            return dict(self.win_rates)

    async def clear_cache(self) -> None:
        """Clear win rate cache."""
        async with self._lock:
            self.win_rates.clear()
            self.last_refresh.clear()
