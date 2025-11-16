"""
GPT-5 Priority 7: Modular Agent Skills System

Implements composable skill modules for agents that can be dynamically
loaded, combined, and optimized based on task requirements.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import asyncio
from datetime import datetime
import uuid
from pathlib import Path
import importlib
import inspect


class SkillType(Enum):
    """Categories of agent skills"""
    CODING = "coding"
    TESTING = "testing"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"


class SkillComplexity(Enum):
    """Complexity levels for skills"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class SkillMetrics:
    """Performance metrics for a skill"""
    success_rate: float = 0.0
    execution_time: float = 0.0
    resource_usage: float = 0.0
    quality_score: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0


@dataclass
class SkillContext:
    """Context information for skill execution"""
    task_description: str
    input_data: Any
    environment_state: Dict[str, Any]
    agent_capabilities: Set[str]
    available_resources: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Result of skill execution"""
    success: bool
    output: Any
    execution_time: float
    resource_usage: Dict[str, float]
    quality_metrics: Dict[str, float]
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    """Abstract base class for all modular skills"""

    def __init__(self, skill_id: str, name: str, skill_type: SkillType,
                 complexity: SkillComplexity, version: str = "1.0.0"):
        self.skill_id = skill_id
        self.name = name
        self.skill_type = skill_type
        self.complexity = complexity
        self.version = version
        self.metrics = SkillMetrics()
        self.dependencies: Set[str] = set()
        self.tags: Set[str] = set()
        self.logger = logging.getLogger(f"skill.{skill_id}")
        self.enabled = True
        self.created_at = datetime.now()

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute the skill with given context"""
        pass

    @abstractmethod
    def get_capabilities(self) -> Set[str]:
        """Return set of capabilities this skill provides"""
        pass

    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """Return requirements for this skill to execute"""
        pass

    def can_execute(self, context: SkillContext) -> bool:
        """Check if skill can execute in given context"""
        requirements = self.get_requirements()

        # Check required capabilities
        if "capabilities" in requirements:
            required_caps = set(requirements["capabilities"])
            if not required_caps.issubset(context.agent_capabilities):
                return False

        # Check required resources
        if "resources" in requirements:
            for resource, min_amount in requirements["resources"].items():
                available = context.available_resources.get(resource, 0)
                if available < min_amount:
                    return False

        # Check constraints
        for constraint, value in context.constraints.items():
            if constraint in requirements.get("constraints", {}):
                if not self._check_constraint(constraint, value, requirements["constraints"][constraint]):
                    return False

        return self.enabled

    def _check_constraint(self, constraint: str, value: Any, requirement: Any) -> bool:
        """Check if constraint value meets requirement"""
        if constraint == "max_execution_time":
            return self.metrics.average_execution_time <= requirement
        elif constraint == "min_quality_score":
            return self.metrics.quality_score >= requirement
        elif constraint == "min_success_rate":
            return self.metrics.success_rate >= requirement
        return True

    def update_metrics(self, result: SkillResult) -> None:
        """Update skill performance metrics"""
        self.metrics.usage_count += 1
        self.metrics.last_used = datetime.now()
        self.metrics.total_execution_time += result.execution_time
        self.metrics.average_execution_time = (
            self.metrics.total_execution_time / self.metrics.usage_count
        )

        if result.success:
            # Update success rate with exponential moving average
            alpha = 0.1
            self.metrics.success_rate = (
                alpha * 1.0 + (1 - alpha) * self.metrics.success_rate
            )

        # Update quality score from result metrics
        if "quality_score" in result.quality_metrics:
            alpha = 0.1
            new_quality = result.quality_metrics["quality_score"]
            self.metrics.quality_score = (
                alpha * new_quality + (1 - alpha) * self.metrics.quality_score
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize skill to dictionary"""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "skill_type": self.skill_type.value,
            "complexity": self.complexity.value,
            "version": self.version,
            "capabilities": list(self.get_capabilities()),
            "requirements": self.get_requirements(),
            "dependencies": list(self.dependencies),
            "tags": list(self.tags),
            "enabled": self.enabled,
            "metrics": {
                "success_rate": self.metrics.success_rate,
                "execution_time": self.metrics.execution_time,
                "quality_score": self.metrics.quality_score,
                "usage_count": self.metrics.usage_count,
                "average_execution_time": self.metrics.average_execution_time
            },
            "created_at": self.created_at.isoformat()
        }


class CompositeSkill(BaseSkill):
    """Skill composed of multiple sub-skills"""

    def __init__(self, skill_id: str, name: str, sub_skills: List[BaseSkill],
                 execution_strategy: str = "sequential", **kwargs):
        super().__init__(skill_id, name, SkillType.ANALYSIS,
                        SkillComplexity.ADVANCED, **kwargs)
        self.sub_skills = sub_skills
        self.execution_strategy = execution_strategy  # sequential, parallel, conditional

        # Aggregate capabilities and requirements
        self._aggregate_capabilities()
        self._aggregate_requirements()

    def _aggregate_capabilities(self):
        """Aggregate capabilities from all sub-skills"""
        self._capabilities = set()
        for skill in self.sub_skills:
            self._capabilities.update(skill.get_capabilities())

    def _aggregate_requirements(self):
        """Aggregate requirements from all sub-skills"""
        self._requirements = {
            "capabilities": set(),
            "resources": {},
            "constraints": {}
        }

        for skill in self.sub_skills:
            req = skill.get_requirements()
            if "capabilities" in req:
                self._requirements["capabilities"].update(req["capabilities"])
            if "resources" in req:
                for resource, amount in req["resources"].items():
                    current = self._requirements["resources"].get(resource, 0)
                    self._requirements["resources"][resource] = max(current, amount)

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute composite skill based on strategy"""
        start_time = datetime.now()
        results = []

        try:
            if self.execution_strategy == "sequential":
                results = await self._execute_sequential(context)
            elif self.execution_strategy == "parallel":
                results = await self._execute_parallel(context)
            elif self.execution_strategy == "conditional":
                results = await self._execute_conditional(context)
            else:
                raise ValueError(f"Unknown execution strategy: {self.execution_strategy}")

            # Aggregate results
            success = all(r.success for r in results)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Combine outputs
            combined_output = [r.output for r in results]

            # Aggregate quality metrics
            quality_metrics = {}
            for result in results:
                for metric, value in result.quality_metrics.items():
                    if metric not in quality_metrics:
                        quality_metrics[metric] = []
                    quality_metrics[metric].append(value)

            # Average quality metrics
            for metric, values in quality_metrics.items():
                quality_metrics[metric] = sum(values) / len(values)

            return SkillResult(
                success=success,
                output=combined_output,
                execution_time=execution_time,
                resource_usage={"composite": 1.0},
                quality_metrics=quality_metrics
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SkillResult(
                success=False,
                output=None,
                execution_time=execution_time,
                resource_usage={"composite": 1.0},
                quality_metrics={},
                error_message=str(e)
            )

    async def _execute_sequential(self, context: SkillContext) -> List[SkillResult]:
        """Execute sub-skills sequentially"""
        results = []
        current_context = context

        for skill in self.sub_skills:
            result = await skill.execute(current_context)
            results.append(result)

            # Use output of previous skill as input for next
            if result.success and result.output is not None:
                current_context = SkillContext(
                    task_description=f"Process output from {skill.name}",
                    input_data=result.output,
                    environment_state=current_context.environment_state,
                    agent_capabilities=current_context.agent_capabilities,
                    available_resources=current_context.available_resources,
                    constraints=current_context.constraints
                )

        return results

    async def _execute_parallel(self, context: SkillContext) -> List[SkillResult]:
        """Execute sub-skills in parallel"""
        tasks = [skill.execute(context) for skill in self.sub_skills]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SkillResult(
                    success=False,
                    output=None,
                    execution_time=0.0,
                    resource_usage={},
                    quality_metrics={},
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_conditional(self, context: SkillContext) -> List[SkillResult]:
        """Execute sub-skills based on conditions"""
        results = []

        for skill in self.sub_skills:
            if skill.can_execute(context):
                result = await skill.execute(context)
                results.append(result)

                # Stop if skill fails and is critical
                if not result.success and "critical" in skill.tags:
                    break

        return results

    def get_capabilities(self) -> Set[str]:
        """Return aggregated capabilities"""
        return self._capabilities.copy()

    def get_requirements(self) -> Dict[str, Any]:
        """Return aggregated requirements"""
        return self._requirements.copy()


class SkillRegistry:
    """Registry for managing available skills"""

    def __init__(self):
        self.skills: Dict[str, BaseSkill] = {}
        self.skill_types: Dict[SkillType, List[str]] = {
            skill_type: [] for skill_type in SkillType
        }
        self.logger = logging.getLogger("skill_registry")

    def register_skill(self, skill: BaseSkill) -> bool:
        """Register a new skill"""
        try:
            if skill.skill_id in self.skills:
                self.logger.warning(f"Skill {skill.skill_id} already registered, updating")

            self.skills[skill.skill_id] = skill
            self.skill_types[skill.skill_type].append(skill.skill_id)

            self.logger.info(f"Registered skill: {skill.name} ({skill.skill_id})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register skill {skill.skill_id}: {e}")
            return False

    def unregister_skill(self, skill_id: str) -> bool:
        """Unregister a skill"""
        if skill_id not in self.skills:
            self.logger.warning(f"Skill {skill_id} not found for unregistration")
            return False

        skill = self.skills[skill_id]
        del self.skills[skill_id]
        self.skill_types[skill.skill_type].remove(skill_id)

        self.logger.info(f"Unregistered skill: {skill_id}")
        return True

    def get_skill(self, skill_id: str) -> Optional[BaseSkill]:
        """Get skill by ID"""
        return self.skills.get(skill_id)

    def find_skills_by_type(self, skill_type: SkillType) -> List[BaseSkill]:
        """Find all skills of a specific type"""
        skill_ids = self.skill_types[skill_type]
        return [self.skills[skill_id] for skill_id in skill_ids]

    def find_skills_by_capability(self, capability: str) -> List[BaseSkill]:
        """Find all skills that provide a specific capability"""
        matching_skills = []
        for skill in self.skills.values():
            if capability in skill.get_capabilities():
                matching_skills.append(skill)
        return matching_skills

    def find_skills_for_context(self, context: SkillContext) -> List[BaseSkill]:
        """Find all skills that can execute in given context"""
        compatible_skills = []
        for skill in self.skills.values():
            if skill.can_execute(context):
                compatible_skills.append(skill)
        return compatible_skills

    def get_best_skill(self, context: SkillContext,
                      skill_type: Optional[SkillType] = None) -> Optional[BaseSkill]:
        """Get best skill for context based on metrics"""
        candidate_skills = self.find_skills_for_context(context)

        if skill_type:
            candidate_skills = [s for s in candidate_skills if s.skill_type == skill_type]

        if not candidate_skills:
            return None

        # Score skills based on metrics
        best_skill = None
        best_score = -1

        for skill in candidate_skills:
            score = (
                skill.metrics.success_rate * 0.4 +
                skill.metrics.quality_score * 0.3 +
                (1 / (skill.metrics.average_execution_time + 1)) * 0.2 +
                (skill.metrics.usage_count / 100) * 0.1
            )

            if score > best_score:
                best_score = score
                best_skill = skill

        return best_skill

    def load_skills_from_directory(self, directory: Path) -> int:
        """Load skills from Python files in directory"""
        loaded_count = 0

        for file_path in directory.glob("**/*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                # Import module
                spec = importlib.util.spec_from_file_location(
                    f"skill_{file_path.stem}", file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find skill classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseSkill) and
                        obj != BaseSkill and
                        obj != CompositeSkill):

                        # Instantiate and register skill
                        skill = obj()
                        if self.register_skill(skill):
                            loaded_count += 1

            except Exception as e:
                self.logger.error(f"Failed to load skills from {file_path}: {e}")

        return loaded_count

    def export_skills_catalog(self) -> Dict[str, Any]:
        """Export catalog of all registered skills"""
        catalog = {
            "total_skills": len(self.skills),
            "skills_by_type": {},
            "skills": {}
        }

        for skill_type in SkillType:
            catalog["skills_by_type"][skill_type.value] = len(self.skill_types[skill_type])

        for skill_id, skill in self.skills.items():
            catalog["skills"][skill_id] = skill.to_dict()

        return catalog


class SkillComposer:
    """Composes complex skills from simpler ones"""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.logger = logging.getLogger("skill_composer")

    def compose_skill_for_task(self, task_description: str,
                             context: SkillContext) -> Optional[BaseSkill]:
        """Compose optimal skill combination for task"""
        # Analyze task requirements
        task_capabilities = self._analyze_task_capabilities(task_description)

        # Find candidate skills
        candidate_skills = []
        for capability in task_capabilities:
            skills = self.registry.find_skills_by_capability(capability)
            candidate_skills.extend(skills)

        # Remove duplicates and filter by context
        unique_skills = list(set(candidate_skills))
        compatible_skills = [s for s in unique_skills if s.can_execute(context)]

        if not compatible_skills:
            return None

        # Check if single skill can handle task
        for skill in compatible_skills:
            skill_caps = skill.get_capabilities()
            if set(task_capabilities).issubset(skill_caps):
                return skill

        # Compose composite skill
        return self._create_composite_skill(compatible_skills, task_capabilities, context)

    def _analyze_task_capabilities(self, task_description: str) -> List[str]:
        """Analyze task description to identify required capabilities"""
        # Simple keyword-based analysis (can be enhanced with NLP)
        capabilities = []

        keywords_map = {
            "code": ["code_generation", "code_analysis"],
            "test": ["test_generation", "test_execution"],
            "debug": ["error_analysis", "debugging"],
            "analyze": ["data_analysis", "performance_analysis"],
            "optimize": ["code_optimization", "performance_optimization"],
            "validate": ["validation", "verification"],
            "document": ["documentation_generation"],
            "refactor": ["code_refactoring"]
        }

        task_lower = task_description.lower()
        for keyword, caps in keywords_map.items():
            if keyword in task_lower:
                capabilities.extend(caps)

        return capabilities

    def _create_composite_skill(self, skills: List[BaseSkill],
                              required_capabilities: List[str],
                              context: SkillContext) -> CompositeSkill:
        """Create composite skill from available skills"""
        # Order skills by dependencies and capabilities
        ordered_skills = self._order_skills_by_dependencies(skills)

        # Determine execution strategy
        strategy = self._determine_execution_strategy(ordered_skills, context)

        # Create composite skill
        composite_id = f"composite_{uuid.uuid4().hex[:8]}"
        composite_name = f"Composite skill for {', '.join(required_capabilities[:3])}"

        composite = CompositeSkill(
            skill_id=composite_id,
            name=composite_name,
            sub_skills=ordered_skills,
            execution_strategy=strategy
        )

        # Register composite skill
        self.registry.register_skill(composite)

        return composite

    def _order_skills_by_dependencies(self, skills: List[BaseSkill]) -> List[BaseSkill]:
        """Order skills based on their dependencies"""
        # Simple dependency ordering (can be enhanced with topological sort)
        coding_skills = [s for s in skills if s.skill_type == SkillType.CODING]
        testing_skills = [s for s in skills if s.skill_type == SkillType.TESTING]
        debugging_skills = [s for s in skills if s.skill_type == SkillType.DEBUGGING]
        analysis_skills = [s for s in skills if s.skill_type == SkillType.ANALYSIS]
        other_skills = [s for s in skills if s.skill_type not in
                       [SkillType.CODING, SkillType.TESTING, SkillType.DEBUGGING, SkillType.ANALYSIS]]

        # Typical order: analysis -> coding -> testing -> debugging
        ordered = analysis_skills + coding_skills + testing_skills + debugging_skills + other_skills
        return ordered

    def _determine_execution_strategy(self, skills: List[BaseSkill],
                                    context: SkillContext) -> str:
        """Determine best execution strategy for skills"""
        # Check if skills have dependencies
        has_dependencies = any(skill.dependencies for skill in skills)

        # Check resource constraints
        max_parallel = context.constraints.get("max_parallel_skills", 3)

        if has_dependencies or len(skills) > max_parallel:
            return "sequential"
        elif len(skills) <= 3:
            return "parallel"
        else:
            return "conditional"


# Global skill registry instance
skill_registry = SkillRegistry()
skill_composer = SkillComposer(skill_registry)