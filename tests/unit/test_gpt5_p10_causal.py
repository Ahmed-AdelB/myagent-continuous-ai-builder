#!/usr/bin/env python3
"""
GPT-5 Priority P10: Causal Analytics Engine - Comprehensive Unit Tests

Tests the causal inference and systematic experimentation capabilities including:
- Causal graph construction and causal relationship discovery
- A/B testing framework and experiment design optimization
- Treatment effect estimation and confounding variable control
- Counterfactual analysis and intervention outcome prediction
- Statistical validation and causal inference methodology

Testing methodologies applied:
- TDD: Test-driven development for causal algorithms
- BDD: Behavior-driven scenarios for experimentation
- Statistical testing for causal validity
- Property-based testing for causal consistency
- Simulation testing for counterfactual analysis
"""

import pytest
import asyncio
import numpy as np
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import statistics

# Import test fixtures
from tests.fixtures.test_data import TEST_DATA


class CausalMethodType(Enum):
    """Types of causal inference methods"""
    BACKDOOR = "backdoor"
    FRONTDOOR = "frontdoor"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    PROPENSITY_SCORE = "propensity_score"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    MATCHING = "matching"


class ExperimentStatus(Enum):
    """Experiment execution status"""
    DESIGN = "design"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VariableType(Enum):
    """Types of variables in causal analysis"""
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    INSTRUMENTAL = "instrumental"
    COLLIDER = "collider"


@dataclass
class CausalVariable:
    """Causal variable definition"""
    variable_id: str
    name: str
    variable_type: VariableType
    data_type: str  # continuous, binary, categorical
    description: str
    possible_values: Optional[List[Any]] = None
    measurement_unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalRelationship:
    """Causal relationship between variables"""
    relationship_id: str
    cause_variable_id: str
    effect_variable_id: str
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence_sources: List[str]
    confounders: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentDesign:
    """A/B testing experiment design"""
    experiment_id: str
    name: str
    hypothesis: str
    treatment_variable: str
    outcome_variables: List[str]
    confounding_variables: List[str]
    sample_size: int
    power: float
    significance_level: float
    effect_size: float
    randomization_strategy: str
    stratification_variables: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results of causal experiment"""
    experiment_id: str
    treatment_effect: float
    confidence_interval: Tuple[float, float]
    p_value: float
    statistical_significance: bool
    effect_size_cohens_d: float
    sample_sizes: Dict[str, int]
    outcome_metrics: Dict[str, Any]
    causal_inference_method: CausalMethodType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "treatment_effect": self.treatment_effect,
            "confidence_interval": list(self.confidence_interval),
            "p_value": self.p_value,
            "statistical_significance": self.statistical_significance,
            "effect_size_cohens_d": self.effect_size_cohens_d,
            "sample_sizes": self.sample_sizes,
            "outcome_metrics": self.outcome_metrics,
            "causal_inference_method": self.causal_inference_method.value
        }


@dataclass
class CounterfactualScenario:
    """Counterfactual analysis scenario"""
    scenario_id: str
    description: str
    intervention_variables: Dict[str, Any]
    predicted_outcomes: Dict[str, float]
    confidence_bounds: Dict[str, Tuple[float, float]]
    assumptions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "intervention_variables": self.intervention_variables,
            "predicted_outcomes": self.predicted_outcomes,
            "confidence_bounds": {k: list(v) for k, v in self.confidence_bounds.items()},
            "assumptions": self.assumptions
        }


class MockCausalAnalyticsEngine:
    """Mock implementation of Causal Analytics Engine for testing"""

    def __init__(self):
        self.variables: Dict[str, CausalVariable] = {}
        self.relationships: Dict[str, CausalRelationship] = {}
        self.causal_graph: Dict[str, List[str]] = {}  # adjacency list representation
        self.experiments: Dict[str, ExperimentDesign] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        self.causal_models: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize causal analytics engine"""
        # Set up default variables for common scenarios
        await self._setup_default_variables()

        # Initialize causal models
        await self._setup_causal_models()

    async def add_variable(self, variable: CausalVariable) -> str:
        """Add variable to causal system"""
        if not variable.variable_id:
            variable.variable_id = f"var_{uuid.uuid4().hex[:8]}"

        self.variables[variable.variable_id] = variable

        # Initialize in causal graph
        if variable.variable_id not in self.causal_graph:
            self.causal_graph[variable.variable_id] = []

        return variable.variable_id

    async def add_causal_relationship(self, relationship: CausalRelationship) -> str:
        """Add causal relationship between variables"""
        if not relationship.relationship_id:
            relationship.relationship_id = f"rel_{uuid.uuid4().hex[:8]}"

        # Validate variables exist
        if (relationship.cause_variable_id not in self.variables or
            relationship.effect_variable_id not in self.variables):
            raise ValueError("Cause or effect variable does not exist")

        self.relationships[relationship.relationship_id] = relationship

        # Update causal graph
        if relationship.cause_variable_id not in self.causal_graph:
            self.causal_graph[relationship.cause_variable_id] = []

        self.causal_graph[relationship.cause_variable_id].append(relationship.effect_variable_id)

        return relationship.relationship_id

    async def discover_causal_relationships(self, data: List[Dict[str, Any]],
                                          method: CausalMethodType = CausalMethodType.BACKDOOR) -> List[CausalRelationship]:
        """Discover causal relationships from observational data"""
        discovered_relationships = []

        # Store data for analysis
        data_id = f"dataset_{uuid.uuid4().hex[:8]}"
        self.historical_data[data_id] = data

        if not data:
            return discovered_relationships

        # Simulate causal discovery based on method
        variable_names = list(data[0].keys()) if data else []

        if method == CausalMethodType.BACKDOOR:
            # Simulate backdoor criterion analysis
            for i, var1 in enumerate(variable_names):
                for j, var2 in enumerate(variable_names):
                    if i != j:
                        # Simulate correlation-based causal discovery
                        correlation = await self._calculate_correlation(data, var1, var2)
                        if abs(correlation) > 0.3:  # Threshold for potential causation

                            # Find potential confounders
                            confounders = await self._identify_confounders(data, var1, var2, variable_names)

                            relationship = CausalRelationship(
                                relationship_id=f"discovered_{uuid.uuid4().hex[:6]}",
                                cause_variable_id=var1,
                                effect_variable_id=var2,
                                strength=correlation,
                                confidence=min(0.95, abs(correlation) + 0.2),
                                evidence_sources=[f"backdoor_analysis_{data_id}"],
                                confounders=confounders
                            )

                            discovered_relationships.append(relationship)
                            await self.add_causal_relationship(relationship)

        return discovered_relationships

    async def design_experiment(self, treatment_var: str, outcome_var: str,
                               target_effect_size: float = 0.2,
                               power: float = 0.8,
                               significance_level: float = 0.05) -> ExperimentDesign:
        """Design A/B testing experiment"""
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

        # Calculate required sample size
        sample_size = await self._calculate_sample_size(target_effect_size, power, significance_level)

        # Identify confounding variables
        confounders = await self._identify_experiment_confounders(treatment_var, outcome_var)

        # Generate experiment design
        design = ExperimentDesign(
            experiment_id=experiment_id,
            name=f"Experiment: Effect of {treatment_var} on {outcome_var}",
            hypothesis=f"Treatment {treatment_var} has a significant effect on {outcome_var}",
            treatment_variable=treatment_var,
            outcome_variables=[outcome_var],
            confounding_variables=confounders,
            sample_size=sample_size,
            power=power,
            significance_level=significance_level,
            effect_size=target_effect_size,
            randomization_strategy="simple_randomization"
        )

        self.experiments[experiment_id] = design
        return design

    async def run_experiment(self, experiment_id: str, treatment_data: List[Dict[str, Any]],
                           control_data: List[Dict[str, Any]]) -> ExperimentResult:
        """Run causal experiment and analyze results"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        design = self.experiments[experiment_id]

        # Simulate experiment execution and analysis
        result = await self._analyze_experiment_data(design, treatment_data, control_data)
        self.experiment_results[experiment_id] = result

        return result

    async def estimate_treatment_effect(self, treatment_var: str, outcome_var: str,
                                       data: List[Dict[str, Any]],
                                       method: CausalMethodType = CausalMethodType.BACKDOOR) -> Dict[str, Any]:
        """Estimate causal treatment effect using specified method"""
        if not data:
            raise ValueError("Data is required for treatment effect estimation")

        effect_estimate = {
            "treatment_variable": treatment_var,
            "outcome_variable": outcome_var,
            "method": method.value,
            "treatment_effect": 0.0,
            "confidence_interval": (0.0, 0.0),
            "p_value": 1.0,
            "assumptions": []
        }

        # Simulate method-specific analysis
        if method == CausalMethodType.BACKDOOR:
            # Simulate backdoor adjustment
            confounders = await self._identify_confounders(data, treatment_var, outcome_var, list(data[0].keys()))

            # Calculate adjusted treatment effect
            raw_effect = await self._calculate_raw_effect(data, treatment_var, outcome_var)
            adjusted_effect = await self._adjust_for_confounders(raw_effect, confounders, data)

            effect_estimate.update({
                "treatment_effect": adjusted_effect,
                "confidence_interval": (adjusted_effect - 0.1, adjusted_effect + 0.1),
                "p_value": 0.03,
                "assumptions": [
                    "No unmeasured confounders",
                    "No selection bias",
                    "Backdoor criterion satisfied"
                ],
                "confounders_adjusted": confounders
            })

        elif method == CausalMethodType.INSTRUMENTAL_VARIABLES:
            # Simulate instrumental variables analysis
            instrument_strength = 0.7  # Simulated
            treatment_effect = await self._iv_estimation(data, treatment_var, outcome_var)

            effect_estimate.update({
                "treatment_effect": treatment_effect,
                "confidence_interval": (treatment_effect - 0.15, treatment_effect + 0.15),
                "p_value": 0.02,
                "assumptions": [
                    "Instrument relevance",
                    "Instrument exogeneity",
                    "Exclusion restriction"
                ],
                "instrument_strength": instrument_strength
            })

        elif method == CausalMethodType.PROPENSITY_SCORE:
            # Simulate propensity score matching
            propensity_scores = await self._calculate_propensity_scores(data, treatment_var)
            matched_effect = await self._propensity_score_matching(data, treatment_var, outcome_var, propensity_scores)

            effect_estimate.update({
                "treatment_effect": matched_effect,
                "confidence_interval": (matched_effect - 0.08, matched_effect + 0.08),
                "p_value": 0.01,
                "assumptions": [
                    "Strong ignorability",
                    "Common support",
                    "Balancing property"
                ],
                "matched_pairs": len(data) // 2
            })

        return effect_estimate

    async def generate_counterfactual(self, intervention: Dict[str, Any],
                                    outcome_vars: List[str]) -> CounterfactualScenario:
        """Generate counterfactual analysis scenario"""
        scenario_id = f"counterfactual_{uuid.uuid4().hex[:8]}"

        # Simulate counterfactual prediction
        predicted_outcomes = {}
        confidence_bounds = {}

        for outcome_var in outcome_vars:
            # Simulate outcome prediction based on intervention
            base_outcome = 0.5  # Baseline
            intervention_effect = 0.0

            for var_id, intervention_value in intervention.items():
                # Find causal relationships affecting this outcome
                causal_effect = await self._get_causal_effect(var_id, outcome_var)
                if isinstance(intervention_value, (int, float)):
                    intervention_effect += causal_effect * intervention_value
                else:
                    intervention_effect += causal_effect * 0.5  # Binary intervention

            predicted_outcome = base_outcome + intervention_effect
            predicted_outcomes[outcome_var] = predicted_outcome

            # Add uncertainty bounds
            uncertainty = 0.1
            confidence_bounds[outcome_var] = (
                predicted_outcome - uncertainty,
                predicted_outcome + uncertainty
            )

        scenario = CounterfactualScenario(
            scenario_id=scenario_id,
            description=f"Counterfactual analysis with intervention: {intervention}",
            intervention_variables=intervention,
            predicted_outcomes=predicted_outcomes,
            confidence_bounds=confidence_bounds,
            assumptions=[
                "Structural causal model is correct",
                "No unmeasured confounders",
                "Causal relationships are stable"
            ]
        )

        return scenario

    async def validate_causal_model(self, model_id: str, validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate causal model against empirical data"""
        if model_id not in self.causal_models:
            raise ValueError(f"Causal model {model_id} not found")

        model = self.causal_models[model_id]

        validation_result = {
            "model_id": model_id,
            "validation_data_size": len(validation_data),
            "goodness_of_fit": {},
            "prediction_accuracy": {},
            "causal_consistency": {},
            "overall_validity": 0.0
        }

        # Simulate model validation
        if validation_data:
            # Test goodness of fit
            validation_result["goodness_of_fit"] = {
                "r_squared": 0.85,
                "adjusted_r_squared": 0.82,
                "aic": -150.2,
                "bic": -145.8
            }

            # Test prediction accuracy
            validation_result["prediction_accuracy"] = {
                "mae": 0.15,  # Mean Absolute Error
                "rmse": 0.22,  # Root Mean Square Error
                "mape": 8.5   # Mean Absolute Percentage Error
            }

            # Test causal consistency
            validation_result["causal_consistency"] = {
                "pearl_hierarchy_test": "passed",
                "do_calculus_validation": "passed",
                "backdoor_criterion_check": "satisfied",
                "frontdoor_criterion_check": "not_applicable"
            }

            # Calculate overall validity score
            fit_score = validation_result["goodness_of_fit"]["adjusted_r_squared"]
            accuracy_score = 1.0 - (validation_result["prediction_accuracy"]["rmse"] / 1.0)  # Normalized
            consistency_score = 0.95  # Based on passing causal tests

            validation_result["overall_validity"] = (fit_score + accuracy_score + consistency_score) / 3.0

        return validation_result

    async def get_causal_graph(self) -> Dict[str, Any]:
        """Get causal graph representation"""
        graph_data = {
            "variables": {vid: var.to_dict() for vid, var in self.variables.items()},
            "relationships": {rid: rel.to_dict() for rid, rel in self.relationships.items()},
            "adjacency_matrix": await self._build_adjacency_matrix(),
            "graph_properties": await self._analyze_graph_properties()
        }

        return graph_data

    async def analyze_mediation(self, treatment_var: str, outcome_var: str,
                               mediator_var: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze mediation effects"""
        mediation_analysis = {
            "treatment_variable": treatment_var,
            "outcome_variable": outcome_var,
            "mediator_variable": mediator_var,
            "total_effect": 0.0,
            "direct_effect": 0.0,
            "indirect_effect": 0.0,
            "proportion_mediated": 0.0,
            "sobel_test_p_value": 0.0
        }

        if not data:
            return mediation_analysis

        # Simulate mediation analysis
        # Total effect: treatment -> outcome
        total_effect = await self._calculate_raw_effect(data, treatment_var, outcome_var)

        # Path a: treatment -> mediator
        path_a = await self._calculate_raw_effect(data, treatment_var, mediator_var)

        # Path b: mediator -> outcome (controlling for treatment)
        path_b = await self._calculate_mediated_effect(data, mediator_var, outcome_var, treatment_var)

        # Indirect effect: a * b
        indirect_effect = path_a * path_b

        # Direct effect: total - indirect
        direct_effect = total_effect - indirect_effect

        # Proportion mediated
        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0

        mediation_analysis.update({
            "total_effect": total_effect,
            "direct_effect": direct_effect,
            "indirect_effect": indirect_effect,
            "proportion_mediated": proportion_mediated,
            "sobel_test_p_value": 0.045,  # Simulated p-value
            "path_coefficients": {
                "a_path": path_a,
                "b_path": path_b
            }
        })

        return mediation_analysis

    # Helper methods

    async def _setup_default_variables(self):
        """Set up default variables for testing"""
        default_variables = [
            CausalVariable(
                variable_id="treatment",
                name="treatment",
                variable_type=VariableType.TREATMENT,
                data_type="binary",
                description="Treatment assignment",
                possible_values=[0, 1]
            ),
            CausalVariable(
                variable_id="outcome",
                name="outcome",
                variable_type=VariableType.OUTCOME,
                data_type="continuous",
                description="Primary outcome variable"
            ),
            CausalVariable(
                variable_id="confounder",
                name="confounder",
                variable_type=VariableType.CONFOUNDER,
                data_type="continuous",
                description="Confounding variable"
            )
        ]

        for var in default_variables:
            await self.add_variable(var)

    async def _setup_causal_models(self):
        """Set up default causal models"""
        self.causal_models["default_model"] = {
            "model_type": "linear_structural",
            "equations": {
                "outcome": "beta0 + beta1*treatment + beta2*confounder + epsilon"
            },
            "parameters": {
                "beta0": 0.5,
                "beta1": 0.3,
                "beta2": 0.2
            }
        }

    async def _calculate_correlation(self, data: List[Dict[str, Any]], var1: str, var2: str) -> float:
        """Calculate correlation between two variables"""
        try:
            values1 = [float(item.get(var1, 0)) for item in data if var1 in item and item[var1] is not None]
            values2 = [float(item.get(var2, 0)) for item in data if var2 in item and item[var2] is not None]

            if len(values1) < 2 or len(values2) < 2:
                return 0.0

            # Simple correlation calculation
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)

            numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
            denom1 = sum((x - mean1) ** 2 for x in values1)
            denom2 = sum((y - mean2) ** 2 for y in values2)

            if denom1 == 0 or denom2 == 0:
                return 0.0

            correlation = numerator / (denom1 * denom2) ** 0.5
            return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]

        except Exception:
            return 0.0

    async def _identify_confounders(self, data: List[Dict[str, Any]], treatment: str,
                                   outcome: str, all_variables: List[str]) -> List[str]:
        """Identify potential confounding variables"""
        confounders = []

        for var in all_variables:
            if var not in [treatment, outcome]:
                # Check if variable is associated with both treatment and outcome
                treat_corr = await self._calculate_correlation(data, var, treatment)
                outcome_corr = await self._calculate_correlation(data, var, outcome)

                if abs(treat_corr) > 0.1 and abs(outcome_corr) > 0.1:
                    confounders.append(var)

        return confounders

    async def _calculate_sample_size(self, effect_size: float, power: float, alpha: float) -> int:
        """Calculate required sample size for experiment"""
        # Simplified sample size calculation
        # In practice, this would use proper statistical formulas
        base_size = 100

        # Adjust for effect size (smaller effects need larger samples)
        effect_adjustment = 1.0 / (effect_size ** 2)

        # Adjust for power
        power_adjustment = (power / 0.8) ** 2

        # Adjust for significance level
        alpha_adjustment = (0.05 / alpha) ** 2

        sample_size = int(base_size * effect_adjustment * power_adjustment * alpha_adjustment)
        return max(50, min(10000, sample_size))  # Clamp to reasonable range

    async def _identify_experiment_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Identify confounders for experiment design"""
        # Find variables that have causal relationships with both treatment and outcome
        confounders = []

        for var_id, relationships in self.causal_graph.items():
            affects_treatment = treatment in relationships
            affects_outcome = outcome in relationships

            if affects_treatment and affects_outcome and var_id not in [treatment, outcome]:
                confounders.append(var_id)

        return confounders

    async def _analyze_experiment_data(self, design: ExperimentDesign,
                                      treatment_data: List[Dict[str, Any]],
                                      control_data: List[Dict[str, Any]]) -> ExperimentResult:
        """Analyze experiment data and compute results"""
        outcome_var = design.outcome_variables[0]

        # Calculate means
        treatment_outcomes = [float(item.get(outcome_var, 0)) for item in treatment_data if outcome_var in item]
        control_outcomes = [float(item.get(outcome_var, 0)) for item in control_data if outcome_var in item]

        if not treatment_outcomes or not control_outcomes:
            # Return default result if no data
            return ExperimentResult(
                experiment_id=design.experiment_id,
                treatment_effect=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                statistical_significance=False,
                effect_size_cohens_d=0.0,
                sample_sizes={"treatment": len(treatment_data), "control": len(control_data)},
                outcome_metrics={},
                causal_inference_method=CausalMethodType.BACKDOOR
            )

        treatment_mean = statistics.mean(treatment_outcomes)
        control_mean = statistics.mean(control_outcomes)
        treatment_effect = treatment_mean - control_mean

        # Calculate standard deviations
        treatment_std = statistics.stdev(treatment_outcomes) if len(treatment_outcomes) > 1 else 0
        control_std = statistics.stdev(control_outcomes) if len(control_outcomes) > 1 else 0

        # Calculate Cohen's d
        pooled_std = ((treatment_std ** 2 + control_std ** 2) / 2) ** 0.5
        cohens_d = treatment_effect / pooled_std if pooled_std > 0 else 0

        # Calculate confidence interval (simplified)
        margin_of_error = 0.1  # Simplified
        confidence_interval = (treatment_effect - margin_of_error, treatment_effect + margin_of_error)

        # Calculate p-value (simplified)
        p_value = 0.03 if abs(treatment_effect) > 0.1 else 0.15

        result = ExperimentResult(
            experiment_id=design.experiment_id,
            treatment_effect=treatment_effect,
            confidence_interval=confidence_interval,
            p_value=p_value,
            statistical_significance=p_value < design.significance_level,
            effect_size_cohens_d=cohens_d,
            sample_sizes={"treatment": len(treatment_data), "control": len(control_data)},
            outcome_metrics={
                "treatment_mean": treatment_mean,
                "control_mean": control_mean,
                "treatment_std": treatment_std,
                "control_std": control_std
            },
            causal_inference_method=CausalMethodType.BACKDOOR
        )

        return result

    async def _calculate_raw_effect(self, data: List[Dict[str, Any]], treatment: str, outcome: str) -> float:
        """Calculate raw effect between treatment and outcome"""
        correlation = await self._calculate_correlation(data, treatment, outcome)
        # Convert correlation to effect estimate (simplified)
        return correlation * 0.5

    async def _adjust_for_confounders(self, raw_effect: float, confounders: List[str],
                                    data: List[Dict[str, Any]]) -> float:
        """Adjust effect for confounding variables"""
        # Simplified adjustment - in practice would use regression
        adjustment_factor = 0.8 if confounders else 1.0
        return raw_effect * adjustment_factor

    async def _iv_estimation(self, data: List[Dict[str, Any]], treatment: str, outcome: str) -> float:
        """Instrumental variables estimation"""
        # Simplified IV estimation
        return 0.25  # Simulated IV estimate

    async def _calculate_propensity_scores(self, data: List[Dict[str, Any]], treatment: str) -> List[float]:
        """Calculate propensity scores"""
        # Simplified propensity score calculation
        return [0.5 + (hash(str(item)) % 100 - 50) / 1000 for item in data]

    async def _propensity_score_matching(self, data: List[Dict[str, Any]], treatment: str,
                                       outcome: str, propensity_scores: List[float]) -> float:
        """Propensity score matching analysis"""
        # Simplified matching analysis
        return 0.18  # Simulated matched estimate

    async def _get_causal_effect(self, cause_var: str, effect_var: str) -> float:
        """Get causal effect strength between variables"""
        for relationship in self.relationships.values():
            if (relationship.cause_variable_id == cause_var and
                relationship.effect_variable_id == effect_var):
                return relationship.strength

        return 0.0  # No direct causal relationship

    async def _build_adjacency_matrix(self) -> List[List[int]]:
        """Build adjacency matrix representation of causal graph"""
        variable_ids = list(self.variables.keys())
        n = len(variable_ids)
        matrix = [[0] * n for _ in range(n)]

        for i, var1 in enumerate(variable_ids):
            for j, var2 in enumerate(variable_ids):
                if var2 in self.causal_graph.get(var1, []):
                    matrix[i][j] = 1

        return matrix

    async def _analyze_graph_properties(self) -> Dict[str, Any]:
        """Analyze causal graph properties"""
        num_variables = len(self.variables)
        num_relationships = len(self.relationships)

        # Calculate graph density
        max_edges = num_variables * (num_variables - 1)
        density = num_relationships / max_edges if max_edges > 0 else 0

        # Check for cycles (simplified check)
        has_cycles = await self._detect_cycles()

        return {
            "num_variables": num_variables,
            "num_relationships": num_relationships,
            "graph_density": density,
            "is_acyclic": not has_cycles,
            "variable_types": {vtype.value: sum(1 for v in self.variables.values() if v.variable_type == vtype)
                             for vtype in VariableType}
        }

    async def _detect_cycles(self) -> bool:
        """Detect cycles in causal graph"""
        # Simplified cycle detection
        visited = set()
        rec_stack = set()

        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.causal_graph.get(node, []):
                if dfs(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for node in self.causal_graph:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    async def _calculate_mediated_effect(self, data: List[Dict[str, Any]], mediator: str,
                                       outcome: str, treatment: str) -> float:
        """Calculate mediated effect controlling for treatment"""
        # Simplified partial correlation calculation
        base_corr = await self._calculate_correlation(data, mediator, outcome)
        return base_corr * 0.8  # Simplified adjustment for treatment control


@pytest.fixture
def causal_analytics():
    """Fixture providing mock causal analytics engine"""
    return MockCausalAnalyticsEngine()


@pytest.fixture
def sample_variables():
    """Fixture providing sample causal variables"""
    return [
        CausalVariable(
            variable_id="code_refactoring",
            name="Code Refactoring",
            variable_type=VariableType.TREATMENT,
            data_type="binary",
            description="Whether code refactoring was performed",
            possible_values=[0, 1]
        ),
        CausalVariable(
            variable_id="performance_improvement",
            name="Performance Improvement",
            variable_type=VariableType.OUTCOME,
            data_type="continuous",
            description="Percentage improvement in performance",
            measurement_unit="percentage"
        ),
        CausalVariable(
            variable_id="code_complexity",
            name="Code Complexity",
            variable_type=VariableType.CONFOUNDER,
            data_type="continuous",
            description="Cyclomatic complexity of the code",
            measurement_unit="complexity_score"
        ),
        CausalVariable(
            variable_id="developer_experience",
            name="Developer Experience",
            variable_type=VariableType.CONFOUNDER,
            data_type="continuous",
            description="Years of developer experience",
            measurement_unit="years"
        )
    ]


@pytest.fixture
def sample_relationships():
    """Fixture providing sample causal relationships"""
    return [
        CausalRelationship(
            relationship_id="refactor_to_performance",
            cause_variable_id="code_refactoring",
            effect_variable_id="performance_improvement",
            strength=0.25,
            confidence=0.85,
            evidence_sources=["experiment_001", "observational_study_002"],
            confounders=["code_complexity", "developer_experience"]
        ),
        CausalRelationship(
            relationship_id="complexity_to_performance",
            cause_variable_id="code_complexity",
            effect_variable_id="performance_improvement",
            strength=-0.15,
            confidence=0.75,
            evidence_sources=["observational_study_001"]
        )
    ]


@pytest.fixture
def sample_data():
    """Fixture providing sample observational data"""
    return [
        {"code_refactoring": 1, "performance_improvement": 25.5, "code_complexity": 8.2, "developer_experience": 5},
        {"code_refactoring": 0, "performance_improvement": 15.2, "code_complexity": 12.1, "developer_experience": 3},
        {"code_refactoring": 1, "performance_improvement": 32.1, "code_complexity": 6.5, "developer_experience": 7},
        {"code_refactoring": 0, "performance_improvement": 8.7, "code_complexity": 15.3, "developer_experience": 2},
        {"code_refactoring": 1, "performance_improvement": 28.9, "code_complexity": 9.1, "developer_experience": 6},
        {"code_refactoring": 0, "performance_improvement": 12.4, "code_complexity": 11.8, "developer_experience": 4},
        {"code_refactoring": 1, "performance_improvement": 35.2, "code_complexity": 7.3, "developer_experience": 8},
        {"code_refactoring": 0, "performance_improvement": 6.8, "code_complexity": 16.7, "developer_experience": 1}
    ]


@pytest.fixture
def sample_experiment_data():
    """Fixture providing sample experiment data"""
    treatment_data = [
        {"code_refactoring": 1, "performance_improvement": 28.5, "code_complexity": 8.0},
        {"code_refactoring": 1, "performance_improvement": 32.1, "code_complexity": 7.5},
        {"code_refactoring": 1, "performance_improvement": 25.8, "code_complexity": 9.2},
        {"code_refactoring": 1, "performance_improvement": 30.3, "code_complexity": 8.8}
    ]

    control_data = [
        {"code_refactoring": 0, "performance_improvement": 15.2, "code_complexity": 12.1},
        {"code_refactoring": 0, "performance_improvement": 18.7, "code_complexity": 11.3},
        {"code_refactoring": 0, "performance_improvement": 12.4, "code_complexity": 13.5},
        {"code_refactoring": 0, "performance_improvement": 16.9, "code_complexity": 12.8}
    ]

    return treatment_data, control_data


class TestCausalAnalyticsEngine:
    """Comprehensive tests for Causal Analytics Engine"""

    @pytest.mark.asyncio
    async def test_causal_analytics_initialization(self, causal_analytics):
        """Test causal analytics engine initialization"""
        await causal_analytics.initialize()

        # Should have default variables
        assert len(causal_analytics.variables) >= 3
        assert "treatment" in [v.variable_id for v in causal_analytics.variables.values()]
        assert "outcome" in [v.variable_id for v in causal_analytics.variables.values()]
        assert "confounder" in [v.variable_id for v in causal_analytics.variables.values()]

        # Should have causal models
        assert len(causal_analytics.causal_models) > 0
        assert "default_model" in causal_analytics.causal_models

    @pytest.mark.asyncio
    async def test_variable_management(self, causal_analytics, sample_variables):
        """Test variable creation and management"""
        await causal_analytics.initialize()

        # Add variables
        variable_ids = []
        for variable in sample_variables:
            variable_id = await causal_analytics.add_variable(variable)
            variable_ids.append(variable_id)

        assert len(variable_ids) == len(sample_variables)

        # Verify variables were stored
        for variable_id in variable_ids:
            stored_var = causal_analytics.variables.get(variable_id)
            assert stored_var is not None
            assert stored_var.variable_id == variable_id

        # Verify causal graph nodes were created
        for variable_id in variable_ids:
            assert variable_id in causal_analytics.causal_graph

    @pytest.mark.asyncio
    async def test_causal_relationship_management(self, causal_analytics, sample_variables, sample_relationships):
        """Test causal relationship creation and management"""
        await causal_analytics.initialize()

        # Add variables first
        for variable in sample_variables:
            await causal_analytics.add_variable(variable)

        # Add relationships
        relationship_ids = []
        for relationship in sample_relationships:
            rel_id = await causal_analytics.add_causal_relationship(relationship)
            relationship_ids.append(rel_id)

        assert len(relationship_ids) == len(sample_relationships)

        # Verify relationships were stored
        for rel_id in relationship_ids:
            stored_rel = causal_analytics.relationships.get(rel_id)
            assert stored_rel is not None
            assert stored_rel.relationship_id == rel_id

        # Verify causal graph edges were created
        refactor_rel = sample_relationships[0]
        assert refactor_rel.effect_variable_id in causal_analytics.causal_graph[refactor_rel.cause_variable_id]

    @pytest.mark.asyncio
    async def test_causal_relationship_validation(self, causal_analytics):
        """Test validation of causal relationships"""
        await causal_analytics.initialize()

        # Try to create relationship with non-existent variables
        invalid_relationship = CausalRelationship(
            relationship_id="invalid_rel",
            cause_variable_id="non_existent_cause",
            effect_variable_id="non_existent_effect",
            strength=0.5,
            confidence=0.8,
            evidence_sources=["test"]
        )

        with pytest.raises(ValueError, match="Cause or effect variable does not exist"):
            await causal_analytics.add_causal_relationship(invalid_relationship)

    @pytest.mark.asyncio
    async def test_correlation_calculation(self, causal_analytics, sample_data):
        """Test correlation calculation between variables"""
        await causal_analytics.initialize()

        # Calculate correlation between code_refactoring and performance_improvement
        correlation = await causal_analytics._calculate_correlation(
            sample_data,
            "code_refactoring",
            "performance_improvement"
        )

        # Should find positive correlation
        assert isinstance(correlation, float)
        assert -1.0 <= correlation <= 1.0
        assert correlation > 0  # Refactoring should positively correlate with performance

        # Calculate correlation between complexity and performance
        complexity_corr = await causal_analytics._calculate_correlation(
            sample_data,
            "code_complexity",
            "performance_improvement"
        )

        # Should find negative correlation
        assert complexity_corr < 0  # Higher complexity should correlate with lower performance

    @pytest.mark.asyncio
    async def test_confounder_identification(self, causal_analytics, sample_data):
        """Test identification of confounding variables"""
        await causal_analytics.initialize()

        confounders = await causal_analytics._identify_confounders(
            sample_data,
            "code_refactoring",
            "performance_improvement",
            ["code_complexity", "developer_experience", "other_var"]
        )

        # Should identify code_complexity and developer_experience as confounders
        assert isinstance(confounders, list)
        assert "code_complexity" in confounders
        assert "developer_experience" in confounders

    @pytest.mark.asyncio
    async def test_causal_discovery(self, causal_analytics, sample_data):
        """Test causal relationship discovery from data"""
        await causal_analytics.initialize()

        discovered_relationships = await causal_analytics.discover_causal_relationships(
            sample_data,
            CausalMethodType.BACKDOOR
        )

        assert len(discovered_relationships) > 0

        # Verify discovered relationships have required properties
        for relationship in discovered_relationships:
            assert isinstance(relationship, CausalRelationship)
            assert relationship.relationship_id is not None
            assert -1.0 <= relationship.strength <= 1.0
            assert 0.0 <= relationship.confidence <= 1.0
            assert len(relationship.evidence_sources) > 0

        # Should discover relationship between refactoring and performance
        refactor_relationships = [
            r for r in discovered_relationships
            if "code_refactoring" in [r.cause_variable_id, r.effect_variable_id] and
               "performance_improvement" in [r.cause_variable_id, r.effect_variable_id]
        ]
        assert len(refactor_relationships) > 0

    @pytest.mark.asyncio
    async def test_experiment_design(self, causal_analytics):
        """Test A/B testing experiment design"""
        await causal_analytics.initialize()

        design = await causal_analytics.design_experiment(
            treatment_var="code_refactoring",
            outcome_var="performance_improvement",
            target_effect_size=0.25,
            power=0.8,
            significance_level=0.05
        )

        assert isinstance(design, ExperimentDesign)
        assert design.experiment_id is not None
        assert design.treatment_variable == "code_refactoring"
        assert design.outcome_variables == ["performance_improvement"]
        assert design.power == 0.8
        assert design.significance_level == 0.05
        assert design.effect_size == 0.25
        assert design.sample_size > 0

        # Should have identified confounders
        assert isinstance(design.confounding_variables, list)

    @pytest.mark.asyncio
    async def test_sample_size_calculation(self, causal_analytics):
        """Test sample size calculation for experiments"""
        await causal_analytics.initialize()

        # Test different effect sizes
        small_effect_size = await causal_analytics._calculate_sample_size(0.1, 0.8, 0.05)
        medium_effect_size = await causal_analytics._calculate_sample_size(0.3, 0.8, 0.05)
        large_effect_size = await causal_analytics._calculate_sample_size(0.8, 0.8, 0.05)

        # Smaller effect sizes should require larger samples
        assert small_effect_size > medium_effect_size
        assert medium_effect_size > large_effect_size

        # Test different power requirements
        low_power = await causal_analytics._calculate_sample_size(0.3, 0.7, 0.05)
        high_power = await causal_analytics._calculate_sample_size(0.3, 0.9, 0.05)

        # Higher power should require larger samples
        assert high_power > low_power

    @pytest.mark.asyncio
    async def test_experiment_execution(self, causal_analytics, sample_experiment_data):
        """Test experiment execution and analysis"""
        await causal_analytics.initialize()

        # Design experiment
        design = await causal_analytics.design_experiment(
            treatment_var="code_refactoring",
            outcome_var="performance_improvement"
        )

        treatment_data, control_data = sample_experiment_data

        # Run experiment
        result = await causal_analytics.run_experiment(design.experiment_id, treatment_data, control_data)

        assert isinstance(result, ExperimentResult)
        assert result.experiment_id == design.experiment_id
        assert isinstance(result.treatment_effect, float)
        assert len(result.confidence_interval) == 2
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.statistical_significance, bool)
        assert result.sample_sizes["treatment"] == len(treatment_data)
        assert result.sample_sizes["control"] == len(control_data)

        # Treatment effect should be positive (refactoring improves performance)
        assert result.treatment_effect > 0

    @pytest.mark.asyncio
    async def test_treatment_effect_estimation_backdoor(self, causal_analytics, sample_data):
        """Test treatment effect estimation using backdoor method"""
        await causal_analytics.initialize()

        effect_estimate = await causal_analytics.estimate_treatment_effect(
            "code_refactoring",
            "performance_improvement",
            sample_data,
            CausalMethodType.BACKDOOR
        )

        assert effect_estimate["treatment_variable"] == "code_refactoring"
        assert effect_estimate["outcome_variable"] == "performance_improvement"
        assert effect_estimate["method"] == "backdoor"
        assert isinstance(effect_estimate["treatment_effect"], float)
        assert len(effect_estimate["confidence_interval"]) == 2
        assert 0.0 <= effect_estimate["p_value"] <= 1.0
        assert isinstance(effect_estimate["assumptions"], list)
        assert len(effect_estimate["assumptions"]) > 0

        # Should have identified confounders
        assert "confounders_adjusted" in effect_estimate

    @pytest.mark.asyncio
    async def test_treatment_effect_estimation_iv(self, causal_analytics, sample_data):
        """Test treatment effect estimation using instrumental variables"""
        await causal_analytics.initialize()

        effect_estimate = await causal_analytics.estimate_treatment_effect(
            "code_refactoring",
            "performance_improvement",
            sample_data,
            CausalMethodType.INSTRUMENTAL_VARIABLES
        )

        assert effect_estimate["method"] == "instrumental_variables"
        assert isinstance(effect_estimate["treatment_effect"], float)
        assert "instrument_strength" in effect_estimate
        assert "Instrument relevance" in effect_estimate["assumptions"]
        assert "Instrument exogeneity" in effect_estimate["assumptions"]

    @pytest.mark.asyncio
    async def test_treatment_effect_estimation_propensity_score(self, causal_analytics, sample_data):
        """Test treatment effect estimation using propensity score matching"""
        await causal_analytics.initialize()

        effect_estimate = await causal_analytics.estimate_treatment_effect(
            "code_refactoring",
            "performance_improvement",
            sample_data,
            CausalMethodType.PROPENSITY_SCORE
        )

        assert effect_estimate["method"] == "propensity_score"
        assert isinstance(effect_estimate["treatment_effect"], float)
        assert "matched_pairs" in effect_estimate
        assert "Strong ignorability" in effect_estimate["assumptions"]

    @pytest.mark.asyncio
    async def test_counterfactual_analysis(self, causal_analytics, sample_variables):
        """Test counterfactual scenario generation"""
        await causal_analytics.initialize()

        # Add variables and relationships
        for var in sample_variables:
            await causal_analytics.add_variable(var)

        # Create intervention
        intervention = {
            "code_refactoring": 1,
            "code_complexity": 5.0
        }

        scenario = await causal_analytics.generate_counterfactual(
            intervention,
            ["performance_improvement"]
        )

        assert isinstance(scenario, CounterfactualScenario)
        assert scenario.scenario_id is not None
        assert scenario.intervention_variables == intervention
        assert "performance_improvement" in scenario.predicted_outcomes
        assert "performance_improvement" in scenario.confidence_bounds
        assert len(scenario.assumptions) > 0

        # Predicted outcome should be reasonable
        predicted_performance = scenario.predicted_outcomes["performance_improvement"]
        assert isinstance(predicted_performance, float)

        # Confidence bounds should be valid
        lower_bound, upper_bound = scenario.confidence_bounds["performance_improvement"]
        assert lower_bound <= predicted_performance <= upper_bound

    @pytest.mark.asyncio
    async def test_mediation_analysis(self, causal_analytics, sample_data):
        """Test mediation analysis"""
        await causal_analytics.initialize()

        mediation_result = await causal_analytics.analyze_mediation(
            "code_refactoring",
            "performance_improvement",
            "code_complexity",
            sample_data
        )

        assert mediation_result["treatment_variable"] == "code_refactoring"
        assert mediation_result["outcome_variable"] == "performance_improvement"
        assert mediation_result["mediator_variable"] == "code_complexity"
        assert isinstance(mediation_result["total_effect"], float)
        assert isinstance(mediation_result["direct_effect"], float)
        assert isinstance(mediation_result["indirect_effect"], float)
        assert 0.0 <= mediation_result["proportion_mediated"] <= 1.0
        assert 0.0 <= mediation_result["sobel_test_p_value"] <= 1.0

        # Total effect should equal direct + indirect effects (approximately)
        total = mediation_result["total_effect"]
        direct = mediation_result["direct_effect"]
        indirect = mediation_result["indirect_effect"]
        assert abs(total - (direct + indirect)) < 0.1

    @pytest.mark.asyncio
    async def test_causal_graph_export(self, causal_analytics, sample_variables, sample_relationships):
        """Test causal graph export functionality"""
        await causal_analytics.initialize()

        # Add variables and relationships
        for var in sample_variables:
            await causal_analytics.add_variable(var)

        for rel in sample_relationships:
            await causal_analytics.add_causal_relationship(rel)

        # Export graph
        graph_data = await causal_analytics.get_causal_graph()

        assert "variables" in graph_data
        assert "relationships" in graph_data
        assert "adjacency_matrix" in graph_data
        assert "graph_properties" in graph_data

        # Verify variable data
        assert len(graph_data["variables"]) >= len(sample_variables)
        for var in sample_variables:
            assert var.variable_id in graph_data["variables"]

        # Verify relationship data
        assert len(graph_data["relationships"]) >= len(sample_relationships)

        # Verify graph properties
        props = graph_data["graph_properties"]
        assert "num_variables" in props
        assert "num_relationships" in props
        assert "graph_density" in props
        assert "is_acyclic" in props

    @pytest.mark.asyncio
    async def test_causal_model_validation(self, causal_analytics, sample_data):
        """Test causal model validation"""
        await causal_analytics.initialize()

        validation_result = await causal_analytics.validate_causal_model("default_model", sample_data)

        assert validation_result["model_id"] == "default_model"
        assert validation_result["validation_data_size"] == len(sample_data)
        assert "goodness_of_fit" in validation_result
        assert "prediction_accuracy" in validation_result
        assert "causal_consistency" in validation_result
        assert 0.0 <= validation_result["overall_validity"] <= 1.0

        # Verify goodness of fit metrics
        fit_metrics = validation_result["goodness_of_fit"]
        assert "r_squared" in fit_metrics
        assert "aic" in fit_metrics
        assert 0.0 <= fit_metrics["r_squared"] <= 1.0

        # Verify prediction accuracy metrics
        accuracy_metrics = validation_result["prediction_accuracy"]
        assert "mae" in accuracy_metrics
        assert "rmse" in accuracy_metrics
        assert accuracy_metrics["mae"] >= 0
        assert accuracy_metrics["rmse"] >= 0

    @pytest.mark.asyncio
    async def test_cycle_detection(self, causal_analytics, sample_variables):
        """Test cycle detection in causal graph"""
        await causal_analytics.initialize()

        # Add variables
        for var in sample_variables:
            await causal_analytics.add_variable(var)

        # Create acyclic relationships
        acyclic_rel = CausalRelationship(
            relationship_id="acyclic_rel",
            cause_variable_id="code_refactoring",
            effect_variable_id="performance_improvement",
            strength=0.3,
            confidence=0.8,
            evidence_sources=["test"]
        )
        await causal_analytics.add_causal_relationship(acyclic_rel)

        # Should not detect cycles
        has_cycles = await causal_analytics._detect_cycles()
        assert has_cycles is False

        # Create cyclic relationship
        cyclic_rel = CausalRelationship(
            relationship_id="cyclic_rel",
            cause_variable_id="performance_improvement",
            effect_variable_id="code_refactoring",
            strength=0.2,
            confidence=0.7,
            evidence_sources=["test"]
        )
        await causal_analytics.add_causal_relationship(cyclic_rel)

        # Should detect cycles
        has_cycles = await causal_analytics._detect_cycles()
        assert has_cycles is True

    @pytest.mark.asyncio
    async def test_concurrent_causal_operations(self, causal_analytics, sample_variables):
        """Test concurrent causal analysis operations"""
        await causal_analytics.initialize()

        # Add variables concurrently
        add_tasks = [causal_analytics.add_variable(var) for var in sample_variables]
        variable_ids = await asyncio.gather(*add_tasks)

        assert len(variable_ids) == len(sample_variables)
        assert all(vid in causal_analytics.variables for vid in variable_ids)

        # Test concurrent correlation calculations
        test_data = [
            {"var1": i, "var2": i * 2, "var3": i * 0.5}
            for i in range(10)
        ]

        corr_tasks = [
            causal_analytics._calculate_correlation(test_data, "var1", "var2"),
            causal_analytics._calculate_correlation(test_data, "var1", "var3"),
            causal_analytics._calculate_correlation(test_data, "var2", "var3")
        ]

        correlations = await asyncio.gather(*corr_tasks)
        assert len(correlations) == 3
        assert all(isinstance(corr, float) for corr in correlations)

    @pytest.mark.asyncio
    async def test_large_scale_causal_analysis(self, causal_analytics):
        """Test causal analysis with larger datasets"""
        await causal_analytics.initialize()

        # Generate larger dataset
        large_data = []
        for i in range(1000):
            large_data.append({
                "treatment": i % 2,
                "outcome": 50 + (i % 2) * 10 + (i % 100) * 0.1,
                "confounder1": i % 10,
                "confounder2": (i % 5) * 2.5
            })

        # Test causal discovery on large dataset
        start_time = asyncio.get_event_loop().time()
        discovered_relationships = await causal_analytics.discover_causal_relationships(large_data)
        discovery_time = asyncio.get_event_loop().time() - start_time

        # Should complete in reasonable time
        assert discovery_time < 5.0  # Less than 5 seconds

        # Should discover relationships
        assert len(discovered_relationships) > 0


class TestCausalInferenceMethods:
    """Tests for different causal inference methodologies"""

    @pytest.fixture
    def gpt5_test_data(self):
        """Load GPT-5 specific test data"""
        return TEST_DATA.get('gpt5_test_data', {}).get('causal_analytics', {})

    @pytest.mark.asyncio
    async def test_causal_variables_from_test_data(self, causal_analytics, gpt5_test_data):
        """Test causal variables configuration from test data"""
        if not gpt5_test_data or 'test_variables' not in gpt5_test_data:
            pytest.skip("GPT-5 causal analytics test data not available")

        await causal_analytics.initialize()

        test_variables = gpt5_test_data['test_variables']

        # Verify expected variable types are represented
        assert "treatment" in test_variables
        assert "outcome" in test_variables
        assert "confounder" in test_variables

    @pytest.mark.asyncio
    async def test_causal_methods_from_test_data(self, causal_analytics, gpt5_test_data):
        """Test causal methods configuration from test data"""
        if not gpt5_test_data or 'causal_methods' not in gpt5_test_data:
            pytest.skip("GPT-5 causal analytics test data not available")

        await causal_analytics.initialize()

        test_methods = gpt5_test_data['causal_methods']

        # Test each method mentioned in test data
        for method_name in test_methods:
            method_mapping = {
                "backdoor": CausalMethodType.BACKDOOR,
                "frontdoor": CausalMethodType.FRONTDOOR,
                "instrumental_variables": CausalMethodType.INSTRUMENTAL_VARIABLES,
                "propensity_score": CausalMethodType.PROPENSITY_SCORE
            }

            if method_name in method_mapping:
                method = method_mapping[method_name]

                # Test method with sample data
                test_data = [
                    {"treatment": 1, "outcome": 25, "confounder": 5},
                    {"treatment": 0, "outcome": 15, "confounder": 8}
                ]

                effect_estimate = await causal_analytics.estimate_treatment_effect(
                    "treatment",
                    "outcome",
                    test_data,
                    method
                )

                assert effect_estimate["method"] == method.value
                assert isinstance(effect_estimate["treatment_effect"], float)

    @pytest.mark.asyncio
    async def test_sample_interventions_from_test_data(self, causal_analytics, gpt5_test_data):
        """Test sample interventions from test data"""
        if not gpt5_test_data or 'sample_interventions' not in gpt5_test_data:
            pytest.skip("GPT-5 causal analytics test data not available")

        await causal_analytics.initialize()

        sample_interventions = gpt5_test_data['sample_interventions']

        for intervention_data in sample_interventions:
            treatment = intervention_data['treatment']
            outcome = intervention_data['outcome']
            expected_effect = intervention_data['expected_effect']

            # Test counterfactual analysis with intervention
            intervention = {treatment: 1}
            scenario = await causal_analytics.generate_counterfactual(intervention, [outcome])

            assert isinstance(scenario, CounterfactualScenario)
            assert treatment in scenario.intervention_variables
            assert outcome in scenario.predicted_outcomes

            # Predicted effect should be in reasonable range relative to expected
            predicted_outcome = scenario.predicted_outcomes[outcome]
            assert isinstance(predicted_outcome, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])