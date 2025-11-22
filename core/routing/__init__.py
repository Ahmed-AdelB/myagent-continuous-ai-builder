"""
Routing Module - Intelligent task-fitness based routing.

Implements task-fitness scoring to route tasks to the most suitable agent:
- Capability matching (domain expertise)
- Load balancing (current workload)
- Historical win-rate (past success)

Components:
- AgentCapabilityMatrix: Agent expertise definitions
- LoadTracker: Real-time agent workload tracking
- WinRateTracker: Historical success rate tracking
- TaskFitnessRouter: Main routing engine

Based on: Issue #6 - Enhanced routing with task-fitness scoring
Implementation: Claude (Sonnet 4.5)
"""

from .capability_matrix import AgentCapabilityMatrix, TaskDomain, AgentCapability
from .load_tracker import LoadTracker, AgentLoad
from .win_rate_tracker import WinRateTracker
from .task_fitness_router import TaskFitnessRouter, RoutingDecision

__all__ = [
    'AgentCapabilityMatrix',
    'TaskDomain',
    'AgentCapability',
    'LoadTracker',
    'AgentLoad',
    'WinRateTracker',
    'TaskFitnessRouter',
    'RoutingDecision',
]
