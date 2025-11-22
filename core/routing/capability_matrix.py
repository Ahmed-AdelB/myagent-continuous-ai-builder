"""
Agent Capability Matrix - Domain expertise definitions.

Defines what each agent is good at for intelligent task routing.
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class TaskDomain(Enum):
    """Task domain categories."""
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_IMPLEMENTATION = "code_implementation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DEBUGGING = "debugging"
    SECURITY_AUDIT = "security_audit"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"
    REFACTORING = "refactoring"


@dataclass
class AgentCapability:
    """Agent capability profile."""
    agent_name: str
    expertise: Dict[TaskDomain, float]  # Domain -> expertise score 0.0-1.0

    def get_expertise(self, domain: TaskDomain) -> float:
        """Get expertise score for domain (0.0 if not defined)."""
        return self.expertise.get(domain, 0.0)


class AgentCapabilityMatrix:
    """
    Defines agent capabilities for task routing.

    Expertise scores:
    - 1.0: Expert (primary domain)
    - 0.7-0.9: Proficient
    - 0.4-0.6: Capable
    - 0.1-0.3: Basic
    - 0.0: Not capable
    """

    def __init__(self):
        """Initialize capability matrix with default agent profiles."""
        self.capabilities: Dict[str, AgentCapability] = {
            "claude": AgentCapability(
                agent_name="claude",
                expertise={
                    TaskDomain.REQUIREMENTS_ANALYSIS: 1.0,  # Expert
                    TaskDomain.ARCHITECTURE_DESIGN: 1.0,    # Expert
                    TaskDomain.DOCUMENTATION: 0.9,          # Proficient
                    TaskDomain.CODE_REVIEW: 0.8,            # Proficient
                    TaskDomain.TESTING: 0.7,                # Proficient
                    TaskDomain.CODE_IMPLEMENTATION: 0.6,    # Capable
                    TaskDomain.DEBUGGING: 0.6,              # Capable
                    TaskDomain.SECURITY_AUDIT: 0.8,         # Proficient
                    TaskDomain.OPTIMIZATION: 0.7,           # Proficient
                    TaskDomain.REFACTORING: 0.8,            # Proficient
                }
            ),

            "codex": AgentCapability(
                agent_name="codex",
                expertise={
                    TaskDomain.CODE_IMPLEMENTATION: 1.0,    # Expert
                    TaskDomain.REFACTORING: 1.0,            # Expert
                    TaskDomain.DEBUGGING: 0.9,              # Proficient
                    TaskDomain.OPTIMIZATION: 0.9,           # Proficient
                    TaskDomain.TESTING: 0.8,                # Proficient
                    TaskDomain.CODE_REVIEW: 0.7,            # Proficient
                    TaskDomain.ARCHITECTURE_DESIGN: 0.5,    # Capable
                    TaskDomain.REQUIREMENTS_ANALYSIS: 0.3,  # Basic
                    TaskDomain.DOCUMENTATION: 0.4,          # Capable
                    TaskDomain.SECURITY_AUDIT: 0.6,         # Capable
                }
            ),

            "gemini": AgentCapability(
                agent_name="gemini",
                expertise={
                    TaskDomain.SECURITY_AUDIT: 1.0,         # Expert
                    TaskDomain.CODE_REVIEW: 1.0,            # Expert
                    TaskDomain.ARCHITECTURE_DESIGN: 0.9,    # Proficient
                    TaskDomain.TESTING: 0.8,                # Proficient
                    TaskDomain.REQUIREMENTS_ANALYSIS: 0.8,  # Proficient
                    TaskDomain.DEBUGGING: 0.7,              # Proficient
                    TaskDomain.OPTIMIZATION: 0.8,           # Proficient
                    TaskDomain.CODE_IMPLEMENTATION: 0.6,    # Capable
                    TaskDomain.DOCUMENTATION: 0.7,          # Proficient
                    TaskDomain.REFACTORING: 0.7,            # Proficient
                }
            ),
        }

    def get_capability(self, agent_name: str) -> AgentCapability:
        """Get capability profile for agent."""
        if agent_name not in self.capabilities:
            # Return default low-capability profile
            return AgentCapability(
                agent_name=agent_name,
                expertise={domain: 0.1 for domain in TaskDomain}
            )
        return self.capabilities[agent_name]

    def get_expertise_score(self, agent_name: str, domain: TaskDomain) -> float:
        """Get expertise score for agent in specific domain."""
        capability = self.get_capability(agent_name)
        return capability.get_expertise(domain)

    def get_top_agents(self, domain: TaskDomain, k: int = 3) -> List[tuple]:
        """
        Get top k agents for domain.

        Returns:
            List of (agent_name, expertise_score) tuples sorted by score
        """
        scores = [
            (agent_name, self.get_expertise_score(agent_name, domain))
            for agent_name in self.capabilities.keys()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def add_agent(self, capability: AgentCapability) -> None:
        """Add or update agent capability profile."""
        self.capabilities[capability.agent_name] = capability

    def update_expertise(
        self,
        agent_name: str,
        domain: TaskDomain,
        new_score: float
    ) -> None:
        """Update expertise score for agent in domain."""
        if agent_name in self.capabilities:
            self.capabilities[agent_name].expertise[domain] = max(0.0, min(1.0, new_score))
