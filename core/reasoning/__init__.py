"""
Cross-Agent Reasoning Layer - GPT-5 Priority 2
Prevents logical conflicts and ensures reasoning consistency across all agents.
"""

from .cross_agent_coordinator import (
    CrossAgentCoordinator,
    ReasoningContext,
    ReasoningConflict,
    ConsensusDecision,
    ReasoningConflictType,
    ConflictSeverity
)

__all__ = [
    'CrossAgentCoordinator',
    'ReasoningContext',
    'ReasoningConflict',
    'ConsensusDecision',
    'ReasoningConflictType',
    'ConflictSeverity'
]