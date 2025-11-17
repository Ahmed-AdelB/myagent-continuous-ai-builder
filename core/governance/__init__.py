"""
Governance Module

GPT-5 Recommended System Governance Components
Implements safe continuous operation controls
"""

from .meta_governor import MetaGovernorAgent, IterationHealth, GovernanceThresholds

__all__ = [
    'MetaGovernorAgent',
    'IterationHealth',
    'GovernanceThresholds'
]