"""
MyAgent AI Agents Module

Enhanced with GPT-5 Recommended Modular Skills System
"""

from .base_agent import PersistentAgent
from .coder_agent import CoderAgent
from .tester_agent import TesterAgent
from .debugger_agent import DebuggerAgent
from .architect_agent import ArchitectAgent
from .analyzer_agent import AnalyzerAgent
from .ui_refiner_agent import UIRefinerAgent
from .modular_skills import (
    BaseSkill,
    CompositeSkill,
    SkillRegistry,
    SkillComposer,
    SkillType,
    SkillComplexity,
    SkillContext,
    SkillResult,
    SkillMetrics,
    skill_registry,
    skill_composer
)
from .example_skills import (
    CodeGenerationSkill,
    TestGenerationSkill,
    CodeAnalysisSkill,
    DebuggingSkill,
    OptimizationSkill
)

__all__ = [
    "PersistentAgent",
    "CoderAgent",
    "TesterAgent",
    "DebuggerAgent",
    "ArchitectAgent",
    "AnalyzerAgent",
    "UIRefinerAgent",
    # Modular Skills System
    "BaseSkill",
    "CompositeSkill",
    "SkillRegistry",
    "SkillComposer",
    "SkillType",
    "SkillComplexity",
    "SkillContext",
    "SkillResult",
    "SkillMetrics",
    "skill_registry",
    "skill_composer",
    # Example Skills
    "CodeGenerationSkill",
    "TestGenerationSkill",
    "CodeAnalysisSkill",
    "DebuggingSkill",
    "OptimizationSkill"
]