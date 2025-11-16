"""
Adaptive Prompt Optimization Engine - GPT-5 Priority 3
Maximizes GPT-5 efficiency through dynamic prompt optimization and learning.
"""

import asyncio
import json
import time
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics
import re

from ..observability.telemetry_engine import get_telemetry, LogLevel

class PromptOptimizationStrategy(Enum):
    A_B_TESTING = "a_b_testing"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CONTEXT_ADAPTATION = "context_adaptation"
    PERFORMANCE_TUNING = "performance_tuning"
    SEMANTIC_ENHANCEMENT = "semantic_enhancement"

class PromptCategory(Enum):
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    EXPLANATION = "explanation"
    CREATIVE = "creative"

@dataclass
class PromptMetrics:
    """Performance metrics for prompt evaluation"""
    prompt_id: str
    success_rate: float
    avg_response_time_ms: float
    avg_token_count: int
    avg_quality_score: float
    total_uses: int
    error_rate: float
    user_satisfaction: float
    coherence_score: float
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt_id': self.prompt_id,
            'success_rate': self.success_rate,
            'avg_response_time_ms': self.avg_response_time_ms,
            'avg_token_count': self.avg_token_count,
            'avg_quality_score': self.avg_quality_score,
            'total_uses': self.total_uses,
            'error_rate': self.error_rate,
            'user_satisfaction': self.user_satisfaction,
            'coherence_score': self.coherence_score,
            'relevance_score': self.relevance_score
        }

@dataclass
class PromptTemplate:
    """Adaptive prompt template with optimization metadata"""
    template_id: str
    category: PromptCategory
    base_template: str
    dynamic_variables: List[str]
    optimization_tags: List[str]
    performance_history: List[PromptMetrics]
    last_updated: datetime
    version: int
    confidence_level: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'template_id': self.template_id,
            'category': self.category.value,
            'base_template': self.base_template,
            'dynamic_variables': self.dynamic_variables,
            'optimization_tags': self.optimization_tags,
            'performance_history': [m.to_dict() for m in self.performance_history],
            'last_updated': self.last_updated.isoformat(),
            'version': self.version,
            'confidence_level': self.confidence_level
        }

@dataclass
class OptimizationExperiment:
    """A/B testing experiment for prompt optimization"""
    experiment_id: str
    category: PromptCategory
    control_prompt: PromptTemplate
    variant_prompts: List[PromptTemplate]
    start_time: datetime
    end_time: Optional[datetime]
    sample_size_per_variant: int
    current_samples: Dict[str, int]
    results: Dict[str, PromptMetrics]
    winner: Optional[str]
    confidence_interval: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'category': self.category.value,
            'control_prompt': self.control_prompt.to_dict(),
            'variant_prompts': [v.to_dict() for v in self.variant_prompts],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'sample_size_per_variant': self.sample_size_per_variant,
            'current_samples': self.current_samples,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'winner': self.winner,
            'confidence_interval': self.confidence_interval
        }

@dataclass
class ContextualOptimization:
    """Context-specific prompt optimization"""
    context_hash: str
    context_features: Dict[str, Any]
    optimized_template: PromptTemplate
    adaptation_strategy: str
    performance_gain: float
    last_used: datetime

class AdaptivePromptOptimizer:
    """
    Adaptive Prompt Optimization Engine

    Continuously optimizes prompts for maximum GPT-5 efficiency through:
    - Real-time performance tracking and adaptation
    - A/B testing for systematic prompt improvement
    - Context-aware prompt generation
    - Reinforcement learning from user feedback
    - Semantic enhancement based on successful patterns
    """

    def __init__(self, telemetry=None):
        self.telemetry = telemetry or get_telemetry()
        self.telemetry.register_component('adaptive_prompt_optimizer')

        # Prompt management
        self.prompt_templates = {}  # template_id -> PromptTemplate
        self.prompt_variants = defaultdict(list)  # category -> List[template_id]
        self.active_experiments = {}  # experiment_id -> OptimizationExperiment

        # Performance tracking
        self.performance_history = deque(maxlen=10000)
        self.contextual_optimizations = {}  # context_hash -> ContextualOptimization
        self.successful_patterns = defaultdict(list)

        # Configuration
        self.optimization_enabled = True
        self.min_samples_for_optimization = 50
        self.confidence_threshold = 0.95
        self.adaptation_learning_rate = 0.1

        # Threading
        self._lock = threading.Lock()
        self._optimization_active = False
        self._optimization_thread = None

        # Optimization strategies
        self.optimization_strategies = {
            PromptOptimizationStrategy.A_B_TESTING: self._run_ab_testing,
            PromptOptimizationStrategy.REINFORCEMENT_LEARNING: self._run_reinforcement_learning,
            PromptOptimizationStrategy.CONTEXT_ADAPTATION: self._run_context_adaptation,
            PromptOptimizationStrategy.PERFORMANCE_TUNING: self._run_performance_tuning,
            PromptOptimizationStrategy.SEMANTIC_ENHANCEMENT: self._run_semantic_enhancement
        }

        # Performance tracking
        self.optimization_metrics = {
            'prompts_optimized': 0,
            'experiments_completed': 0,
            'performance_improvements': 0,
            'adaptations_made': 0
        }

        self._initialize_default_templates()

    async def start(self):
        """Start adaptive prompt optimization"""
        self.telemetry.log_info("Starting adaptive prompt optimizer", 'adaptive_prompt_optimizer')

        self._optimization_active = True
        self._optimization_thread = threading.Thread(target=self._optimization_loop)
        self._optimization_thread.daemon = True
        self._optimization_thread.start()

        self.telemetry.log_info("Adaptive prompt optimizer started", 'adaptive_prompt_optimizer')

    async def stop(self):
        """Stop optimization"""
        self.telemetry.log_info("Stopping adaptive prompt optimizer", 'adaptive_prompt_optimizer')

        self._optimization_active = False
        if self._optimization_thread:
            self._optimization_thread.join(timeout=10)

        # Finalize active experiments
        await self._finalize_active_experiments()

        self.telemetry.log_info("Adaptive prompt optimizer stopped", 'adaptive_prompt_optimizer')

    def get_optimized_prompt(self, category: PromptCategory, context: Dict[str, Any] = None,
                           agent_id: str = None) -> Tuple[str, str]:
        """Get optimized prompt for given category and context"""

        # Generate context hash for contextual optimization
        context_hash = self._generate_context_hash(context or {})

        # Check for contextual optimization
        if context_hash in self.contextual_optimizations:
            contextual_opt = self.contextual_optimizations[context_hash]
            template = contextual_opt.optimized_template
            contextual_opt.last_used = datetime.now(timezone.utc)

            self.telemetry.log_debug(
                f"Using contextual optimization for {category.value}",
                'adaptive_prompt_optimizer',
                {'context_hash': context_hash, 'template_id': template.template_id}
            )

            return self._render_template(template, context or {}), template.template_id

        # Get best performing template for category
        template = self._get_best_template(category)
        if not template:
            # Fallback to default template
            template = self._create_default_template(category)

        # Check if we should start A/B testing
        if self._should_start_experiment(category):
            self._start_ab_experiment(category)

        return self._render_template(template, context or {}), template.template_id

    def record_prompt_performance(self, template_id: str, response_time_ms: float,
                                success: bool, quality_score: float = None,
                                token_count: int = None, user_satisfaction: float = None):
        """Record performance data for prompt optimization"""

        performance_data = {
            'template_id': template_id,
            'timestamp': datetime.now(timezone.utc),
            'response_time_ms': response_time_ms,
            'success': success,
            'quality_score': quality_score or 0.0,
            'token_count': token_count or 0,
            'user_satisfaction': user_satisfaction or 0.0
        }

        self.performance_history.append(performance_data)

        # Update template metrics
        self._update_template_metrics(template_id, performance_data)

        # Check for optimization opportunities
        self._check_optimization_triggers(template_id)

        self.telemetry.log_debug(
            f"Recorded performance for template {template_id}",
            'adaptive_prompt_optimizer',
            performance_data
        )

    def create_prompt_variant(self, base_template_id: str, modifications: Dict[str, Any],
                            variant_name: str = None) -> str:
        """Create a new prompt variant for testing"""

        with self._lock:
            base_template = self.prompt_templates.get(base_template_id)
            if not base_template:
                raise ValueError(f"Base template {base_template_id} not found")

        variant_id = variant_name or f"{base_template_id}_variant_{uuid.uuid4().hex[:8]}"

        # Apply modifications to base template
        variant_template = self._apply_template_modifications(base_template, modifications)
        variant_template.template_id = variant_id
        variant_template.version = base_template.version + 1

        with self._lock:
            self.prompt_templates[variant_id] = variant_template
            self.prompt_variants[base_template.category].append(variant_id)

        self.telemetry.log_info(
            f"Created prompt variant: {variant_id}",
            'adaptive_prompt_optimizer',
            {'base_template': base_template_id, 'modifications': list(modifications.keys())}
        )

        return variant_id

    def start_optimization_experiment(self, category: PromptCategory,
                                    strategy: PromptOptimizationStrategy = PromptOptimizationStrategy.A_B_TESTING,
                                    sample_size: int = 100) -> str:
        """Start an optimization experiment for a category"""

        experiment_id = str(uuid.uuid4())

        if strategy == PromptOptimizationStrategy.A_B_TESTING:
            experiment = self._create_ab_experiment(experiment_id, category, sample_size)
        else:
            # Other strategies would be implemented here
            raise NotImplementedError(f"Strategy {strategy.value} not yet implemented")

        with self._lock:
            self.active_experiments[experiment_id] = experiment

        self.telemetry.log_info(
            f"Started optimization experiment: {experiment_id}",
            'adaptive_prompt_optimizer',
            {
                'category': category.value,
                'strategy': strategy.value,
                'sample_size': sample_size
            }
        )

        return experiment_id

    def get_optimization_insights(self, category: PromptCategory = None) -> Dict[str, Any]:
        """Get optimization insights and recommendations"""

        insights = {
            'overall_performance': self._calculate_overall_performance(),
            'category_performance': {},
            'optimization_opportunities': [],
            'active_experiments': len(self.active_experiments),
            'total_templates': len(self.prompt_templates)
        }

        # Category-specific insights
        categories = [category] if category else list(PromptCategory)
        for cat in categories:
            insights['category_performance'][cat.value] = self._analyze_category_performance(cat)

        # Identify optimization opportunities
        insights['optimization_opportunities'] = self._identify_optimization_opportunities()

        # Recent improvements
        insights['recent_improvements'] = self._get_recent_improvements()

        return insights

    def get_template_analytics(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analytics for a specific template"""

        with self._lock:
            template = self.prompt_templates.get(template_id)
            if not template:
                return None

        # Calculate performance metrics
        recent_performance = self._get_recent_performance(template_id, days=7)

        analytics = {
            'template_info': template.to_dict(),
            'performance_summary': self._calculate_template_performance_summary(template_id),
            'performance_trends': self._analyze_performance_trends(template_id),
            'optimization_suggestions': self._generate_optimization_suggestions(template_id),
            'usage_patterns': self._analyze_usage_patterns(template_id)
        }

        return analytics

    # Internal methods

    def _initialize_default_templates(self):
        """Initialize default prompt templates for each category"""
        default_templates = {
            PromptCategory.CODE_GENERATION: {
                'template': """You are an expert software engineer. Generate high-quality, well-documented code for the following requirement:

Requirement: {requirement}

Context: {context}

Please provide:
1. Clean, efficient code with appropriate error handling
2. Inline comments explaining complex logic
3. Proper variable naming and structure
4. Any necessary imports or dependencies

Code:""",
                'variables': ['requirement', 'context'],
                'tags': ['coding', 'engineering', 'documentation']
            },
            PromptCategory.REASONING: {
                'template': """You are an expert reasoner. Analyze the following problem step by step using clear logical reasoning:

Problem: {problem}

Context: {context}

Please provide:
1. Problem breakdown and key components
2. Step-by-step logical analysis
3. Consideration of multiple perspectives
4. Clear conclusion with supporting evidence

Analysis:""",
                'variables': ['problem', 'context'],
                'tags': ['reasoning', 'analysis', 'logic']
            },
            PromptCategory.DEBUGGING: {
                'template': """You are an expert debugger. Help diagnose and fix the following issue:

Error/Issue: {error_description}

Code Context: {code_context}

Environment: {environment}

Please provide:
1. Root cause analysis
2. Step-by-step debugging approach
3. Specific fix recommendations
4. Prevention strategies for future

Solution:""",
                'variables': ['error_description', 'code_context', 'environment'],
                'tags': ['debugging', 'troubleshooting', 'fix']
            }
        }

        for category, template_data in default_templates.items():
            template_id = f"default_{category.value}"

            template = PromptTemplate(
                template_id=template_id,
                category=category,
                base_template=template_data['template'],
                dynamic_variables=template_data['variables'],
                optimization_tags=template_data['tags'],
                performance_history=[],
                last_updated=datetime.now(timezone.utc),
                version=1,
                confidence_level=0.5
            )

            self.prompt_templates[template_id] = template
            self.prompt_variants[category].append(template_id)

    def _get_best_template(self, category: PromptCategory) -> Optional[PromptTemplate]:
        """Get best performing template for category"""

        with self._lock:
            candidate_ids = self.prompt_variants[category]
            if not candidate_ids:
                return None

            candidates = [self.prompt_templates[tid] for tid in candidate_ids]

        # Score templates based on recent performance
        scored_templates = []
        for template in candidates:
            score = self._calculate_template_score(template)
            scored_templates.append((score, template))

        if not scored_templates:
            return None

        # Return highest scoring template
        scored_templates.sort(key=lambda x: x[0], reverse=True)
        return scored_templates[0][1]

    def _calculate_template_score(self, template: PromptTemplate) -> float:
        """Calculate composite score for template performance"""

        if not template.performance_history:
            return template.confidence_level

        recent_metrics = template.performance_history[-10:]  # Last 10 uses

        # Weight different metrics
        weights = {
            'success_rate': 0.3,
            'quality_score': 0.25,
            'response_time': 0.2,
            'user_satisfaction': 0.15,
            'coherence_score': 0.1
        }

        total_score = 0.0
        for metrics in recent_metrics:
            score = (
                weights['success_rate'] * (1.0 if metrics.success_rate > 0.9 else metrics.success_rate) +
                weights['quality_score'] * metrics.avg_quality_score +
                weights['response_time'] * (1.0 - min(metrics.avg_response_time_ms / 10000, 1.0)) +
                weights['user_satisfaction'] * metrics.user_satisfaction +
                weights['coherence_score'] * metrics.coherence_score
            )
            total_score += score

        avg_score = total_score / len(recent_metrics)

        # Boost for templates with more usage (confidence)
        usage_boost = min(len(template.performance_history) / 100, 0.1)

        return min(avg_score + usage_boost, 1.0)

    def _render_template(self, template: PromptTemplate, context: Dict[str, Any]) -> str:
        """Render template with context variables"""

        rendered = template.base_template

        # Replace template variables
        for var in template.dynamic_variables:
            placeholder = f"{{{var}}}"
            value = context.get(var, f"[{var} not provided]")
            rendered = rendered.replace(placeholder, str(value))

        return rendered

    def _update_template_metrics(self, template_id: str, performance_data: Dict[str, Any]):
        """Update template performance metrics"""

        with self._lock:
            template = self.prompt_templates.get(template_id)
            if not template:
                return

            # Calculate new metrics
            metrics = PromptMetrics(
                prompt_id=template_id,
                success_rate=1.0 if performance_data['success'] else 0.0,
                avg_response_time_ms=performance_data['response_time_ms'],
                avg_token_count=performance_data['token_count'],
                avg_quality_score=performance_data['quality_score'],
                total_uses=1,
                error_rate=0.0 if performance_data['success'] else 1.0,
                user_satisfaction=performance_data['user_satisfaction'],
                coherence_score=performance_data['quality_score'],  # Simplified
                relevance_score=performance_data['quality_score']   # Simplified
            )

            # Add to performance history
            template.performance_history.append(metrics)

            # Keep only recent history
            if len(template.performance_history) > 100:
                template.performance_history = template.performance_history[-50:]

            template.last_updated = datetime.now(timezone.utc)

    def _should_start_experiment(self, category: PromptCategory) -> bool:
        """Determine if we should start an A/B experiment"""

        # Check if we already have an active experiment for this category
        with self._lock:
            for experiment in self.active_experiments.values():
                if experiment.category == category:
                    return False

        # Check if we have enough data to justify an experiment
        template_ids = self.prompt_variants[category]
        if len(template_ids) < 2:
            return False

        total_uses = 0
        with self._lock:
            for template_id in template_ids:
                template = self.prompt_templates.get(template_id)
                if template:
                    total_uses += len(template.performance_history)

        return total_uses >= self.min_samples_for_optimization

    def _start_ab_experiment(self, category: PromptCategory):
        """Start A/B testing experiment"""

        experiment_id = str(uuid.uuid4())

        # Get templates for testing
        template_ids = self.prompt_variants[category]
        if len(template_ids) < 2:
            return

        with self._lock:
            templates = [self.prompt_templates[tid] for tid in template_ids[:4]]  # Max 4 variants

        # Create experiment
        experiment = OptimizationExperiment(
            experiment_id=experiment_id,
            category=category,
            control_prompt=templates[0],
            variant_prompts=templates[1:],
            start_time=datetime.now(timezone.utc),
            end_time=None,
            sample_size_per_variant=50,
            current_samples={t.template_id: 0 for t in templates},
            results={},
            winner=None,
            confidence_interval=0.95
        )

        with self._lock:
            self.active_experiments[experiment_id] = experiment

        self.telemetry.log_info(
            f"Started A/B experiment for {category.value}",
            'adaptive_prompt_optimizer',
            {'experiment_id': experiment_id, 'variants': len(templates)}
        )

    def _create_ab_experiment(self, experiment_id: str, category: PromptCategory,
                            sample_size: int) -> OptimizationExperiment:
        """Create A/B testing experiment"""

        # Get existing templates or create variants
        template_ids = self.prompt_variants[category]

        with self._lock:
            if len(template_ids) >= 2:
                templates = [self.prompt_templates[tid] for tid in template_ids[:4]]
            else:
                # Create variants of the existing template
                base_template = self.prompt_templates[template_ids[0]] if template_ids else None
                if not base_template:
                    base_template = self._create_default_template(category)

                templates = [base_template]
                templates.extend(self._create_automatic_variants(base_template))

        return OptimizationExperiment(
            experiment_id=experiment_id,
            category=category,
            control_prompt=templates[0],
            variant_prompts=templates[1:],
            start_time=datetime.now(timezone.utc),
            end_time=None,
            sample_size_per_variant=sample_size,
            current_samples={t.template_id: 0 for t in templates},
            results={},
            winner=None,
            confidence_interval=0.95
        )

    def _create_automatic_variants(self, base_template: PromptTemplate) -> List[PromptTemplate]:
        """Create automatic variants of a template for testing"""

        variants = []

        # Variant 1: More structured approach
        structured_template = base_template.base_template.replace(
            "Please provide:",
            "Please provide a detailed response with the following structure:"
        )

        variant1 = PromptTemplate(
            template_id=f"{base_template.template_id}_structured",
            category=base_template.category,
            base_template=structured_template,
            dynamic_variables=base_template.dynamic_variables,
            optimization_tags=base_template.optimization_tags + ['structured'],
            performance_history=[],
            last_updated=datetime.now(timezone.utc),
            version=base_template.version + 1,
            confidence_level=0.5
        )
        variants.append(variant1)

        # Variant 2: More concise approach
        concise_template = base_template.base_template.replace(
            "Please provide:",
            "Provide a concise response including:"
        )

        variant2 = PromptTemplate(
            template_id=f"{base_template.template_id}_concise",
            category=base_template.category,
            base_template=concise_template,
            dynamic_variables=base_template.dynamic_variables,
            optimization_tags=base_template.optimization_tags + ['concise'],
            performance_history=[],
            last_updated=datetime.now(timezone.utc),
            version=base_template.version + 1,
            confidence_level=0.5
        )
        variants.append(variant2)

        return variants

    def _create_default_template(self, category: PromptCategory) -> PromptTemplate:
        """Create a default template for category"""

        basic_templates = {
            PromptCategory.CODE_GENERATION: "Generate code for: {requirement}",
            PromptCategory.REASONING: "Analyze: {problem}",
            PromptCategory.ANALYSIS: "Analyze: {data}",
            PromptCategory.PLANNING: "Create plan for: {objective}",
            PromptCategory.DEBUGGING: "Debug: {error_description}",
            PromptCategory.EXPLANATION: "Explain: {concept}",
            PromptCategory.CREATIVE: "Create: {creative_prompt}"
        }

        template_id = f"default_{category.value}_{uuid.uuid4().hex[:8]}"

        return PromptTemplate(
            template_id=template_id,
            category=category,
            base_template=basic_templates.get(category, "Process: {input}"),
            dynamic_variables=['requirement', 'problem', 'data', 'objective', 'error_description', 'concept', 'creative_prompt', 'input'],
            optimization_tags=['default'],
            performance_history=[],
            last_updated=datetime.now(timezone.utc),
            version=1,
            confidence_level=0.3
        )

    def _apply_template_modifications(self, base_template: PromptTemplate,
                                   modifications: Dict[str, Any]) -> PromptTemplate:
        """Apply modifications to create a new template variant"""

        modified_template = PromptTemplate(
            template_id="",  # Will be set by caller
            category=base_template.category,
            base_template=base_template.base_template,
            dynamic_variables=base_template.dynamic_variables[:],
            optimization_tags=base_template.optimization_tags[:],
            performance_history=[],
            last_updated=datetime.now(timezone.utc),
            version=base_template.version,
            confidence_level=base_template.confidence_level
        )

        # Apply modifications
        if 'template_text' in modifications:
            modified_template.base_template = modifications['template_text']

        if 'add_variables' in modifications:
            modified_template.dynamic_variables.extend(modifications['add_variables'])

        if 'add_tags' in modifications:
            modified_template.optimization_tags.extend(modifications['add_tags'])

        if 'prepend_instruction' in modifications:
            modified_template.base_template = (
                modifications['prepend_instruction'] + "\n\n" + modified_template.base_template
            )

        if 'append_instruction' in modifications:
            modified_template.base_template = (
                modified_template.base_template + "\n\n" + modifications['append_instruction']
            )

        return modified_template

    def _generate_context_hash(self, context: Dict[str, Any]) -> str:
        """Generate hash for context to enable contextual optimization"""

        # Sort and serialize context for consistent hashing
        sorted_items = sorted(context.items())
        context_str = json.dumps(sorted_items, sort_keys=True)

        return hashlib.md5(context_str.encode()).hexdigest()

    def _check_optimization_triggers(self, template_id: str):
        """Check if template needs optimization"""

        with self._lock:
            template = self.prompt_templates.get(template_id)
            if not template or len(template.performance_history) < 10:
                return

        # Check performance degradation
        recent_metrics = template.performance_history[-5:]
        older_metrics = template.performance_history[-10:-5]

        if len(older_metrics) >= 5:
            recent_avg_score = statistics.mean([m.avg_quality_score for m in recent_metrics])
            older_avg_score = statistics.mean([m.avg_quality_score for m in older_metrics])

            if recent_avg_score < older_avg_score * 0.9:  # 10% degradation
                self._trigger_template_optimization(template_id)

    def _trigger_template_optimization(self, template_id: str):
        """Trigger optimization for underperforming template"""

        with self._lock:
            template = self.prompt_templates.get(template_id)
            if not template:
                return

        self.telemetry.log_info(
            f"Triggering optimization for template: {template_id}",
            'adaptive_prompt_optimizer',
            {'category': template.category.value}
        )

        # Create optimization variants
        variants = self._create_optimization_variants(template)

        for variant in variants:
            self.prompt_templates[variant.template_id] = variant
            self.prompt_variants[template.category].append(variant.template_id)

        # Start experiment if conditions are met
        if self._should_start_experiment(template.category):
            self._start_ab_experiment(template.category)

    def _create_optimization_variants(self, template: PromptTemplate) -> List[PromptTemplate]:
        """Create optimization variants for underperforming template"""

        variants = []

        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns(template)

        # Create targeted improvements
        if 'clarity' in failure_patterns:
            clarity_variant = self._create_clarity_improved_variant(template)
            variants.append(clarity_variant)

        if 'specificity' in failure_patterns:
            specificity_variant = self._create_specificity_improved_variant(template)
            variants.append(specificity_variant)

        if 'context' in failure_patterns:
            context_variant = self._create_context_improved_variant(template)
            variants.append(context_variant)

        return variants

    def _analyze_failure_patterns(self, template: PromptTemplate) -> List[str]:
        """Analyze patterns in template failures"""

        patterns = []

        # Simple pattern analysis based on performance metrics
        recent_metrics = template.performance_history[-20:]

        avg_quality = statistics.mean([m.avg_quality_score for m in recent_metrics])
        avg_satisfaction = statistics.mean([m.user_satisfaction for m in recent_metrics])

        if avg_quality < 0.6:
            patterns.append('clarity')

        if avg_satisfaction < 0.7:
            patterns.append('specificity')

        if statistics.mean([m.coherence_score for m in recent_metrics]) < 0.7:
            patterns.append('context')

        return patterns

    def _create_clarity_improved_variant(self, template: PromptTemplate) -> PromptTemplate:
        """Create variant with improved clarity"""

        modifications = {
            'prepend_instruction': "Please provide a clear, step-by-step response.",
            'add_tags': ['clarity_improved']
        }

        variant = self._apply_template_modifications(template, modifications)
        variant.template_id = f"{template.template_id}_clarity_{uuid.uuid4().hex[:8]}"

        return variant

    def _create_specificity_improved_variant(self, template: PromptTemplate) -> PromptTemplate:
        """Create variant with improved specificity"""

        modifications = {
            'append_instruction': "Be specific and include concrete examples where applicable.",
            'add_tags': ['specificity_improved']
        }

        variant = self._apply_template_modifications(template, modifications)
        variant.template_id = f"{template.template_id}_specific_{uuid.uuid4().hex[:8]}"

        return variant

    def _create_context_improved_variant(self, template: PromptTemplate) -> PromptTemplate:
        """Create variant with improved context handling"""

        modifications = {
            'prepend_instruction': "Consider the full context provided and address all relevant aspects.",
            'add_tags': ['context_improved']
        }

        variant = self._apply_template_modifications(template, modifications)
        variant.template_id = f"{template.template_id}_context_{uuid.uuid4().hex[:8]}"

        return variant

    def _optimization_loop(self):
        """Background optimization loop"""

        while self._optimization_active:
            try:
                # Check active experiments
                self._process_active_experiments()

                # Run scheduled optimizations
                self._run_scheduled_optimizations()

                # Update optimization metrics
                self._update_optimization_metrics()

            except Exception as e:
                self.telemetry.log_error(
                    f"Error in optimization loop: {e}",
                    'adaptive_prompt_optimizer'
                )

            time.sleep(60)  # Run every minute

    def _process_active_experiments(self):
        """Process and evaluate active experiments"""

        with self._lock:
            experiments_to_complete = []

            for experiment_id, experiment in self.active_experiments.items():
                # Check if experiment is ready for evaluation
                total_samples = sum(experiment.current_samples.values())
                required_samples = experiment.sample_size_per_variant * (1 + len(experiment.variant_prompts))

                if total_samples >= required_samples:
                    experiments_to_complete.append(experiment_id)

        # Complete ready experiments
        for experiment_id in experiments_to_complete:
            self._complete_experiment(experiment_id)

    def _complete_experiment(self, experiment_id: str):
        """Complete and analyze experiment results"""

        with self._lock:
            experiment = self.active_experiments.get(experiment_id)
            if not experiment:
                return

            # Calculate results for each variant
            all_templates = [experiment.control_prompt] + experiment.variant_prompts

            for template in all_templates:
                metrics = self._calculate_experiment_metrics(template, experiment.start_time)
                experiment.results[template.template_id] = metrics

            # Determine winner
            winner_id = self._determine_experiment_winner(experiment)
            experiment.winner = winner_id
            experiment.end_time = datetime.now(timezone.utc)

            # Remove from active experiments
            del self.active_experiments[experiment_id]

        # Apply optimization results
        self._apply_experiment_results(experiment)

        self.optimization_metrics['experiments_completed'] += 1

        self.telemetry.log_info(
            f"Completed optimization experiment: {experiment_id}",
            'adaptive_prompt_optimizer',
            {
                'category': experiment.category.value,
                'winner': experiment.winner,
                'total_variants': len(experiment.results)
            }
        )

    def _calculate_experiment_metrics(self, template: PromptTemplate,
                                    start_time: datetime) -> PromptMetrics:
        """Calculate metrics for template during experiment period"""

        # Filter performance data for experiment period
        experiment_data = [
            perf for perf in self.performance_history
            if (perf['template_id'] == template.template_id and
                perf['timestamp'] >= start_time)
        ]

        if not experiment_data:
            return PromptMetrics(
                prompt_id=template.template_id,
                success_rate=0.0,
                avg_response_time_ms=0.0,
                avg_token_count=0,
                avg_quality_score=0.0,
                total_uses=0,
                error_rate=1.0,
                user_satisfaction=0.0,
                coherence_score=0.0,
                relevance_score=0.0
            )

        # Calculate aggregate metrics
        success_count = sum(1 for d in experiment_data if d['success'])
        total_uses = len(experiment_data)

        return PromptMetrics(
            prompt_id=template.template_id,
            success_rate=success_count / total_uses if total_uses > 0 else 0.0,
            avg_response_time_ms=statistics.mean([d['response_time_ms'] for d in experiment_data]),
            avg_token_count=int(statistics.mean([d['token_count'] for d in experiment_data])),
            avg_quality_score=statistics.mean([d['quality_score'] for d in experiment_data]),
            total_uses=total_uses,
            error_rate=(total_uses - success_count) / total_uses if total_uses > 0 else 1.0,
            user_satisfaction=statistics.mean([d['user_satisfaction'] for d in experiment_data]),
            coherence_score=statistics.mean([d['quality_score'] for d in experiment_data]),
            relevance_score=statistics.mean([d['quality_score'] for d in experiment_data])
        )

    def _determine_experiment_winner(self, experiment: OptimizationExperiment) -> Optional[str]:
        """Determine winning template from experiment results"""

        if not experiment.results:
            return None

        # Score each template
        template_scores = {}

        for template_id, metrics in experiment.results.items():
            # Composite score based on multiple factors
            score = (
                metrics.success_rate * 0.3 +
                metrics.avg_quality_score * 0.25 +
                (1.0 - min(metrics.avg_response_time_ms / 10000, 1.0)) * 0.2 +
                metrics.user_satisfaction * 0.15 +
                metrics.coherence_score * 0.1
            )
            template_scores[template_id] = score

        # Find highest scoring template
        winner_id = max(template_scores, key=template_scores.get)
        winner_score = template_scores[winner_id]

        # Check if improvement is statistically significant
        control_score = template_scores.get(experiment.control_prompt.template_id, 0)

        if winner_score > control_score * 1.05:  # At least 5% improvement
            return winner_id

        return experiment.control_prompt.template_id  # No significant improvement

    def _apply_experiment_results(self, experiment: OptimizationExperiment):
        """Apply results of completed experiment"""

        if not experiment.winner:
            return

        winner_template_id = experiment.winner

        # Update template confidence and promotion
        with self._lock:
            winner_template = self.prompt_templates.get(winner_template_id)
            if winner_template:
                winner_template.confidence_level = min(winner_template.confidence_level + 0.1, 1.0)

                # If winner is not the control, promote it
                if winner_template_id != experiment.control_prompt.template_id:
                    self.optimization_metrics['performance_improvements'] += 1

                    self.telemetry.log_info(
                        f"Promoted optimized template: {winner_template_id}",
                        'adaptive_prompt_optimizer',
                        {
                            'category': experiment.category.value,
                            'improvement': 'significant'
                        }
                    )

    def _run_scheduled_optimizations(self):
        """Run scheduled optimization strategies"""

        # Run different strategies periodically
        strategies_to_run = []

        current_time = datetime.now(timezone.utc)

        # Context adaptation every 30 minutes
        if not hasattr(self, '_last_context_optimization') or \
           (current_time - self._last_context_optimization).total_seconds() > 1800:
            strategies_to_run.append(PromptOptimizationStrategy.CONTEXT_ADAPTATION)
            self._last_context_optimization = current_time

        # Semantic enhancement every hour
        if not hasattr(self, '_last_semantic_optimization') or \
           (current_time - self._last_semantic_optimization).total_seconds() > 3600:
            strategies_to_run.append(PromptOptimizationStrategy.SEMANTIC_ENHANCEMENT)
            self._last_semantic_optimization = current_time

        # Run selected strategies
        for strategy in strategies_to_run:
            try:
                optimizer_func = self.optimization_strategies[strategy]
                optimizer_func()
            except Exception as e:
                self.telemetry.log_error(
                    f"Error running optimization strategy {strategy.value}: {e}",
                    'adaptive_prompt_optimizer'
                )

    # Optimization strategy implementations

    def _run_ab_testing(self):
        """Run A/B testing optimization"""
        # Already implemented in main workflow
        pass

    def _run_reinforcement_learning(self):
        """Run reinforcement learning optimization"""
        # Placeholder for future implementation
        self.telemetry.log_debug("Reinforcement learning optimization not yet implemented", 'adaptive_prompt_optimizer')

    def _run_context_adaptation(self):
        """Run context adaptation optimization"""

        # Analyze context patterns
        context_patterns = self._analyze_context_patterns()

        for context_hash, pattern_data in context_patterns.items():
            if pattern_data['usage_count'] >= 10:  # Minimum usage threshold
                optimized_template = self._create_context_specific_template(pattern_data)
                if optimized_template:
                    self.contextual_optimizations[context_hash] = ContextualOptimization(
                        context_hash=context_hash,
                        context_features=pattern_data['features'],
                        optimized_template=optimized_template,
                        adaptation_strategy='context_specific',
                        performance_gain=pattern_data.get('expected_gain', 0.1),
                        last_used=datetime.now(timezone.utc)
                    )

                    self.optimization_metrics['adaptations_made'] += 1

    def _run_performance_tuning(self):
        """Run performance tuning optimization"""

        # Identify slow-performing templates
        slow_templates = []

        with self._lock:
            for template in self.prompt_templates.values():
                if template.performance_history:
                    recent_metrics = template.performance_history[-10:]
                    avg_response_time = statistics.mean([m.avg_response_time_ms for m in recent_metrics])

                    if avg_response_time > 5000:  # 5 seconds threshold
                        slow_templates.append(template)

        # Create optimized variants for slow templates
        for template in slow_templates:
            performance_variant = self._create_performance_optimized_variant(template)
            if performance_variant:
                self.prompt_templates[performance_variant.template_id] = performance_variant
                self.prompt_variants[template.category].append(performance_variant.template_id)

    def _run_semantic_enhancement(self):
        """Run semantic enhancement optimization"""

        # Analyze successful prompt patterns
        successful_patterns = self._extract_successful_patterns()

        # Apply patterns to underperforming templates
        for pattern in successful_patterns:
            underperforming_templates = self._find_underperforming_templates(pattern['category'])

            for template in underperforming_templates:
                enhanced_template = self._apply_semantic_enhancement(template, pattern)
                if enhanced_template:
                    self.prompt_templates[enhanced_template.template_id] = enhanced_template
                    self.prompt_variants[template.category].append(enhanced_template.template_id)

    # Helper methods for optimization strategies

    def _analyze_context_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze patterns in context usage"""

        patterns = {}

        # Group performance data by context patterns
        for perf_data in list(self.performance_history)[-1000:]:  # Recent data
            # Simple pattern extraction (in real implementation would be more sophisticated)
            context_signature = f"template_{perf_data['template_id']}"

            if context_signature not in patterns:
                patterns[context_signature] = {
                    'usage_count': 0,
                    'success_rate': 0,
                    'features': {},
                    'performance_history': []
                }

            patterns[context_signature]['usage_count'] += 1
            patterns[context_signature]['performance_history'].append(perf_data)

        # Calculate aggregated metrics for each pattern
        for pattern_data in patterns.values():
            if pattern_data['usage_count'] > 0:
                successes = sum(1 for p in pattern_data['performance_history'] if p['success'])
                pattern_data['success_rate'] = successes / pattern_data['usage_count']

        return patterns

    def _create_context_specific_template(self, pattern_data: Dict[str, Any]) -> Optional[PromptTemplate]:
        """Create template optimized for specific context pattern"""

        # Simple implementation - in reality would use more sophisticated analysis
        if pattern_data['success_rate'] < 0.8:
            return None

        # Create enhanced template based on successful pattern
        base_template_id = pattern_data['performance_history'][0]['template_id']

        with self._lock:
            base_template = self.prompt_templates.get(base_template_id)
            if not base_template:
                return None

        modifications = {
            'prepend_instruction': "Based on the specific context provided:",
            'add_tags': ['context_optimized']
        }

        enhanced_template = self._apply_template_modifications(base_template, modifications)
        enhanced_template.template_id = f"{base_template_id}_context_{uuid.uuid4().hex[:8]}"
        enhanced_template.confidence_level = min(base_template.confidence_level + 0.15, 1.0)

        return enhanced_template

    def _create_performance_optimized_variant(self, template: PromptTemplate) -> Optional[PromptTemplate]:
        """Create performance-optimized variant"""

        modifications = {
            'prepend_instruction': "Provide a direct, efficient response.",
            'add_tags': ['performance_optimized']
        }

        optimized_template = self._apply_template_modifications(template, modifications)
        optimized_template.template_id = f"{template.template_id}_perf_{uuid.uuid4().hex[:8]}"

        return optimized_template

    def _extract_successful_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns from successful prompts"""

        patterns = []

        # Analyze high-performing templates
        with self._lock:
            high_performers = []
            for template in self.prompt_templates.values():
                if template.performance_history:
                    score = self._calculate_template_score(template)
                    if score > 0.8:
                        high_performers.append(template)

        # Extract common patterns
        for template in high_performers:
            pattern = {
                'category': template.category,
                'optimization_tags': template.optimization_tags,
                'template_features': self._extract_template_features(template),
                'performance_score': self._calculate_template_score(template)
            }
            patterns.append(pattern)

        return patterns

    def _extract_template_features(self, template: PromptTemplate) -> Dict[str, Any]:
        """Extract features from template for pattern analysis"""

        features = {
            'has_structure': 'structure' in template.base_template.lower() or 'step' in template.base_template.lower(),
            'has_examples': 'example' in template.base_template.lower(),
            'is_specific': len(template.base_template.split()) > 50,
            'has_context': 'context' in template.base_template.lower(),
            'variable_count': len(template.dynamic_variables)
        }

        return features

    def _find_underperforming_templates(self, category: PromptCategory) -> List[PromptTemplate]:
        """Find templates that could benefit from enhancement"""

        underperforming = []

        with self._lock:
            for template_id in self.prompt_variants[category]:
                template = self.prompt_templates.get(template_id)
                if template and template.performance_history:
                    score = self._calculate_template_score(template)
                    if score < 0.6:
                        underperforming.append(template)

        return underperforming

    def _apply_semantic_enhancement(self, template: PromptTemplate,
                                  pattern: Dict[str, Any]) -> Optional[PromptTemplate]:
        """Apply semantic enhancement based on successful pattern"""

        modifications = {}

        # Apply enhancements based on successful pattern features
        pattern_features = pattern['template_features']

        if pattern_features.get('has_structure') and 'structure' not in template.base_template.lower():
            modifications['append_instruction'] = "Structure your response clearly with numbered points."

        if pattern_features.get('has_examples') and 'example' not in template.base_template.lower():
            modifications['append_instruction'] = "Include relevant examples to illustrate your points."

        if not modifications:
            return None

        enhanced_template = self._apply_template_modifications(template, modifications)
        enhanced_template.template_id = f"{template.template_id}_enhanced_{uuid.uuid4().hex[:8]}"
        enhanced_template.optimization_tags.append('semantically_enhanced')

        return enhanced_template

    def _update_optimization_metrics(self):
        """Update optimization performance metrics"""

        # Emit optimization metrics to telemetry
        for metric_name, value in self.optimization_metrics.items():
            self.telemetry.set_gauge(
                f'optimization_{metric_name}',
                value,
                component='adaptive_prompt_optimizer'
            )

        # Calculate optimization effectiveness
        with self._lock:
            total_templates = len(self.prompt_templates)
            active_experiments = len(self.active_experiments)

        effectiveness_score = min(
            self.optimization_metrics['performance_improvements'] / max(total_templates, 1) * 100,
            100
        )

        self.telemetry.set_gauge(
            'optimization_effectiveness_score',
            effectiveness_score,
            component='adaptive_prompt_optimizer'
        )

    async def _finalize_active_experiments(self):
        """Finalize all active experiments on shutdown"""

        with self._lock:
            active_experiment_ids = list(self.active_experiments.keys())

        for experiment_id in active_experiment_ids:
            self._complete_experiment(experiment_id)

        self.telemetry.log_info(
            f"Finalized {len(active_experiment_ids)} active experiments",
            'adaptive_prompt_optimizer'
        )

    # Analysis and insights methods

    def _calculate_overall_performance(self) -> Dict[str, float]:
        """Calculate overall optimization performance"""

        recent_data = list(self.performance_history)[-1000:]

        if not recent_data:
            return {
                'avg_success_rate': 0.0,
                'avg_quality_score': 0.0,
                'avg_response_time_ms': 0.0,
                'optimization_impact': 0.0
            }

        return {
            'avg_success_rate': statistics.mean([1.0 if d['success'] else 0.0 for d in recent_data]),
            'avg_quality_score': statistics.mean([d['quality_score'] for d in recent_data]),
            'avg_response_time_ms': statistics.mean([d['response_time_ms'] for d in recent_data]),
            'optimization_impact': self._calculate_optimization_impact()
        }

    def _calculate_optimization_impact(self) -> float:
        """Calculate the impact of optimization efforts"""

        if len(self.performance_history) < 100:
            return 0.0

        # Compare recent performance to baseline
        recent_data = list(self.performance_history)[-100:]
        baseline_data = list(self.performance_history)[-500:-400] if len(self.performance_history) >= 500 else []

        if not baseline_data:
            return 0.0

        recent_score = statistics.mean([d['quality_score'] for d in recent_data])
        baseline_score = statistics.mean([d['quality_score'] for d in baseline_data])

        improvement = (recent_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0
        return max(0, min(improvement, 100))

    def _analyze_category_performance(self, category: PromptCategory) -> Dict[str, Any]:
        """Analyze performance for specific category"""

        category_data = []

        with self._lock:
            for template_id in self.prompt_variants[category]:
                template = self.prompt_templates.get(template_id)
                if template:
                    for perf in template.performance_history:
                        category_data.append({
                            'template_id': template_id,
                            'success': perf.success_rate > 0.9,
                            'quality_score': perf.avg_quality_score,
                            'response_time_ms': perf.avg_response_time_ms
                        })

        if not category_data:
            return {
                'template_count': 0,
                'avg_performance': 0.0,
                'optimization_opportunities': []
            }

        return {
            'template_count': len(self.prompt_variants[category]),
            'avg_performance': statistics.mean([d['quality_score'] for d in category_data]),
            'success_rate': statistics.mean([1.0 if d['success'] else 0.0 for d in category_data]),
            'avg_response_time': statistics.mean([d['response_time_ms'] for d in category_data]),
            'optimization_opportunities': self._identify_category_opportunities(category)
        }

    def _identify_optimization_opportunities(self) -> List[Dict[str, str]]:
        """Identify optimization opportunities across all templates"""

        opportunities = []

        # Check for categories with low performance
        for category in PromptCategory:
            category_perf = self._analyze_category_performance(category)

            if category_perf['avg_performance'] < 0.7:
                opportunities.append({
                    'type': 'category_improvement',
                    'category': category.value,
                    'description': f"Category {category.value} has low average performance ({category_perf['avg_performance']:.2f})",
                    'priority': 'high' if category_perf['avg_performance'] < 0.5 else 'medium'
                })

        # Check for templates with no recent optimization
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)

        with self._lock:
            for template in self.prompt_templates.values():
                if template.last_updated < cutoff_time:
                    opportunities.append({
                        'type': 'stale_template',
                        'template_id': template.template_id,
                        'description': f"Template {template.template_id} hasn't been optimized recently",
                        'priority': 'low'
                    })

        return opportunities[:10]  # Return top 10 opportunities

    def _identify_category_opportunities(self, category: PromptCategory) -> List[str]:
        """Identify optimization opportunities for specific category"""

        opportunities = []

        # Check if category needs more template variants
        variant_count = len(self.prompt_variants[category])
        if variant_count < 2:
            opportunities.append("Create template variants for A/B testing")

        # Check if category has active experiments
        has_active_experiment = any(
            exp.category == category for exp in self.active_experiments.values()
        )

        if not has_active_experiment and variant_count >= 2:
            opportunities.append("Start A/B testing experiment")

        return opportunities

    def _get_recent_improvements(self) -> List[Dict[str, Any]]:
        """Get recent optimization improvements"""

        improvements = []

        # Check for recently completed experiments
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)

        # This would be populated from a persistent store in a real implementation
        # For now, return placeholder data

        return improvements

    def _get_recent_performance(self, template_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent performance data for template"""

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

        recent_data = []
        for perf in self.performance_history:
            if (perf['template_id'] == template_id and
                perf['timestamp'] >= cutoff_time):
                recent_data.append(perf)

        return recent_data

    def _calculate_template_performance_summary(self, template_id: str) -> Dict[str, float]:
        """Calculate performance summary for template"""

        with self._lock:
            template = self.prompt_templates.get(template_id)
            if not template or not template.performance_history:
                return {}

        recent_metrics = template.performance_history[-50:]  # Last 50 uses

        return {
            'success_rate': statistics.mean([m.success_rate for m in recent_metrics]),
            'avg_quality_score': statistics.mean([m.avg_quality_score for m in recent_metrics]),
            'avg_response_time_ms': statistics.mean([m.avg_response_time_ms for m in recent_metrics]),
            'user_satisfaction': statistics.mean([m.user_satisfaction for m in recent_metrics]),
            'total_uses': len(template.performance_history)
        }

    def _analyze_performance_trends(self, template_id: str) -> Dict[str, str]:
        """Analyze performance trends for template"""

        with self._lock:
            template = self.prompt_templates.get(template_id)
            if not template or len(template.performance_history) < 10:
                return {'trend': 'insufficient_data'}

        recent_metrics = template.performance_history[-10:]
        older_metrics = template.performance_history[-20:-10] if len(template.performance_history) >= 20 else []

        if not older_metrics:
            return {'trend': 'insufficient_data'}

        recent_avg = statistics.mean([m.avg_quality_score for m in recent_metrics])
        older_avg = statistics.mean([m.avg_quality_score for m in older_metrics])

        if recent_avg > older_avg * 1.05:
            return {'trend': 'improving', 'change': f"+{((recent_avg/older_avg - 1) * 100):.1f}%"}
        elif recent_avg < older_avg * 0.95:
            return {'trend': 'declining', 'change': f"{((recent_avg/older_avg - 1) * 100):.1f}%"}
        else:
            return {'trend': 'stable', 'change': '0.0%'}

    def _generate_optimization_suggestions(self, template_id: str) -> List[str]:
        """Generate optimization suggestions for template"""

        suggestions = []

        with self._lock:
            template = self.prompt_templates.get(template_id)
            if not template:
                return suggestions

        # Analyze performance and suggest improvements
        if template.performance_history:
            recent_score = statistics.mean([m.avg_quality_score for m in template.performance_history[-10:]])

            if recent_score < 0.7:
                suggestions.append("Consider adding more specific instructions")
                suggestions.append("Try including examples in the prompt")

            if statistics.mean([m.avg_response_time_ms for m in template.performance_history[-10:]]) > 5000:
                suggestions.append("Simplify prompt to reduce response time")

            if statistics.mean([m.user_satisfaction for m in template.performance_history[-10:]]) < 0.7:
                suggestions.append("Improve prompt clarity and specificity")

        return suggestions

    def _analyze_usage_patterns(self, template_id: str) -> Dict[str, Any]:
        """Analyze usage patterns for template"""

        usage_data = []
        for perf in self.performance_history:
            if perf['template_id'] == template_id:
                usage_data.append({
                    'timestamp': perf['timestamp'],
                    'success': perf['success']
                })

        if not usage_data:
            return {'pattern': 'no_usage'}

        # Simple pattern analysis
        recent_uses = len([u for u in usage_data if u['timestamp'] >= datetime.now(timezone.utc) - timedelta(days=7)])
        total_uses = len(usage_data)

        return {
            'total_uses': total_uses,
            'recent_uses_7d': recent_uses,
            'usage_trend': 'active' if recent_uses > total_uses * 0.3 else 'declining',
            'first_used': min([u['timestamp'] for u in usage_data]).isoformat() if usage_data else None,
            'last_used': max([u['timestamp'] for u in usage_data]).isoformat() if usage_data else None
        }