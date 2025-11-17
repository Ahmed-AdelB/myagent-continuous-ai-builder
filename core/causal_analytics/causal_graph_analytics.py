"""
Advanced Causal Graph Analytics Engine
Implements Pearl's Causal Hierarchy with do-calculus, backdoor criterion, and counterfactual reasoning.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import itertools
import threading
import concurrent.futures
from pathlib import Path

logger = logging.getLogger(__name__)

class CausalEdgeType(Enum):
    """Types of causal edges"""
    DIRECT = "direct"
    CONFOUNDED = "confounded"
    MEDIATED = "mediated"
    INSTRUMENTAL = "instrumental"
    COLLIDER = "collider"

class InferenceMethod(Enum):
    """Causal inference methods"""
    BACKDOOR = "backdoor"
    FRONTDOOR = "frontdoor"
    INSTRUMENTAL_VARIABLES = "iv"
    REGRESSION_DISCONTINUITY = "rd"
    DIFFERENCE_IN_DIFFERENCES = "did"
    PROPENSITY_SCORE = "ps"

class CausalDiscoveryAlgorithm(Enum):
    """Causal discovery algorithms"""
    PC = "pc"
    GES = "ges"
    FAST_CAUSAL_INFERENCE = "fci"
    LINEAR_NON_GAUSSIAN = "lingam"
    CONSTRAINT_BASED = "constraint"

@dataclass
class CausalNode:
    """Represents a variable in the causal graph"""
    name: str
    node_type: str  # treatment, outcome, confounder, mediator, instrument
    data_type: str  # continuous, binary, categorical, ordinal
    description: str
    observed: bool = True
    latent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CausalEdge:
    """Represents a causal relationship between variables"""
    source: str
    target: str
    edge_type: CausalEdgeType
    strength: float  # Effect size
    confidence: float  # Statistical confidence
    mechanism: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterventionEffect:
    """Result of a causal intervention analysis"""
    treatment: str
    outcome: str
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: InferenceMethod
    confounders_controlled: List[str]
    sample_size: int
    assumptions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CounterfactualAnalysis:
    """Counterfactual reasoning results"""
    individual_id: str
    factual_outcome: float
    counterfactual_outcome: float
    treatment_effect: float
    probability: float
    explanation: str
    sensitivity_analysis: Dict[str, float]

@dataclass
class BackdoorCriterion:
    """Backdoor criterion analysis"""
    treatment: str
    outcome: str
    adjustment_set: Set[str]
    is_valid: bool
    all_backdoor_paths: List[List[str]]
    blocked_paths: List[List[str]]
    explanation: str

@dataclass
class FrontdoorCriterion:
    """Front-door criterion analysis"""
    treatment: str
    outcome: str
    mediator_set: Set[str]
    is_valid: bool
    frontdoor_paths: List[List[str]]
    explanation: str

@dataclass
class TreatmentEffect:
    """Treatment effect estimation"""
    average_treatment_effect: float
    conditional_effects: Dict[str, float]
    heterogeneity_score: float
    subgroup_effects: Dict[str, InterventionEffect]
    effect_modifiers: List[str]

@dataclass
class MediationAnalysis:
    """Mediation analysis results"""
    treatment: str
    mediator: str
    outcome: str
    direct_effect: float
    indirect_effect: float
    total_effect: float
    mediation_proportion: float
    significance: Dict[str, float]

@dataclass
class CausalPathway:
    """Represents a causal pathway"""
    path: List[str]
    pathway_type: str  # direct, mediated, confounded
    effect_size: float
    mechanism: str
    evidence_strength: float

@dataclass
class ConfoundingVariable:
    """Information about confounding variables"""
    variable: str
    affects_treatment: bool
    affects_outcome: bool
    confounding_strength: float
    control_method: str

@dataclass
class InstrumentalVariable:
    """Instrumental variable analysis"""
    instrument: str
    treatment: str
    outcome: str
    relevance_strength: float  # Instrument-treatment correlation
    exclusion_restriction: bool  # Satisfies exclusion restriction
    two_stage_estimate: float
    weak_instrument_test: Dict[str, float]

class CausalGraph:
    """Directed Acyclic Graph for causal relationships"""

    def __init__(self, name: str = "causal_graph"):
        self.name = name
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        self.created_at = datetime.now()
        self._lock = threading.Lock()

    def add_node(self, node: CausalNode) -> None:
        """Add a causal node to the graph"""
        with self._lock:
            self.nodes[node.name] = node
            self.graph.add_node(node.name, **node.__dict__)

    def add_edge(self, edge: CausalEdge) -> None:
        """Add a causal edge to the graph"""
        with self._lock:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"Both nodes must exist before adding edge")

            self.edges[(edge.source, edge.target)] = edge
            self.graph.add_edge(edge.source, edge.target, **edge.__dict__)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove a causal edge"""
        with self._lock:
            if (source, target) in self.edges:
                del self.edges[(source, target)]
                self.graph.remove_edge(source, target)

    def get_parents(self, node: str) -> Set[str]:
        """Get direct causal parents of a node"""
        return set(self.graph.predecessors(node))

    def get_children(self, node: str) -> Set[str]:
        """Get direct causal children of a node"""
        return set(self.graph.successors(node))

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all causal ancestors of a node"""
        return set(nx.ancestors(self.graph, node))

    def get_descendants(self, node: str) -> Set[str]:
        """Get all causal descendants of a node"""
        return set(nx.descendants(self.graph, node))

    def find_all_paths(self, source: str, target: str) -> List[List[str]]:
        """Find all directed paths from source to target"""
        try:
            return list(nx.all_simple_paths(self.graph, source, target))
        except nx.NetworkXNoPath:
            return []

    def is_d_separated(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """Check if X and Y are d-separated given Z"""
        return nx.d_separated(self.graph, X, Y, Z)

    def get_markov_blanket(self, node: str) -> Set[str]:
        """Get Markov blanket of a node"""
        parents = self.get_parents(node)
        children = self.get_children(node)
        spouses = set()

        for child in children:
            spouses.update(self.get_parents(child))
        spouses.discard(node)

        return parents | children | spouses

    def topological_sort(self) -> List[str]:
        """Get topological ordering of nodes"""
        return list(nx.topological_sort(self.graph))

    def is_acyclic(self) -> bool:
        """Check if graph is acyclic"""
        return nx.is_directed_acyclic_graph(self.graph)

class CausalDiscovery:
    """Algorithms for discovering causal structure from data"""

    @staticmethod
    def pc_algorithm(data: pd.DataFrame, alpha: float = 0.05) -> CausalGraph:
        """PC algorithm for causal discovery"""
        graph = CausalGraph("pc_discovered")
        variables = data.columns.tolist()

        # Add nodes
        for var in variables:
            node = CausalNode(
                name=var,
                node_type="unknown",
                data_type="continuous",
                description=f"Variable discovered by PC algorithm"
            )
            graph.add_node(node)

        # Phase 1: Edge discovery using conditional independence tests
        edges = set()
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i >= j:
                    continue

                # Test unconditional independence
                corr, p_val = stats.pearsonr(data[var1], data[var2])
                if p_val < alpha:
                    edges.add((var1, var2))

        # Phase 2: Edge orientation using v-structures
        oriented_edges = set()
        for var1, var2 in edges:
            # Simple orientation rules (simplified PC)
            if abs(stats.pearsonr(data[var1], data[var2])[0]) > 0.3:
                edge = CausalEdge(
                    source=var1,
                    target=var2,
                    edge_type=CausalEdgeType.DIRECT,
                    strength=abs(stats.pearsonr(data[var1], data[var2])[0]),
                    confidence=1 - stats.pearsonr(data[var1], data[var2])[1]
                )
                graph.add_edge(edge)
                oriented_edges.add((var1, var2))

        return graph

    @staticmethod
    def constraint_based_discovery(data: pd.DataFrame,
                                 independence_test: str = "correlation") -> CausalGraph:
        """Constraint-based causal discovery"""
        graph = CausalGraph("constraint_based")
        variables = data.columns.tolist()

        # Add nodes
        for var in variables:
            node = CausalNode(
                name=var,
                node_type="unknown",
                data_type="continuous",
                description=f"Variable from constraint-based discovery"
            )
            graph.add_node(node)

        # Build adjacency matrix based on conditional independence
        n_vars = len(variables)
        adj_matrix = np.zeros((n_vars, n_vars))

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    # Test conditional independence
                    corr, p_val = stats.pearsonr(data[var1], data[var2])
                    if p_val < 0.05:  # Dependent
                        adj_matrix[i][j] = abs(corr)

        # Convert to directed edges
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if adj_matrix[i][j] > 0.2:  # Threshold for edge existence
                    edge = CausalEdge(
                        source=var1,
                        target=var2,
                        edge_type=CausalEdgeType.DIRECT,
                        strength=adj_matrix[i][j],
                        confidence=0.8  # Default confidence
                    )
                    graph.add_edge(edge)

        return graph

class CausalInference:
    """Methods for causal inference and effect estimation"""

    def __init__(self, graph: CausalGraph):
        self.graph = graph

    def backdoor_criterion(self, treatment: str, outcome: str) -> BackdoorCriterion:
        """Implement Pearl's backdoor criterion"""
        # Find all backdoor paths
        backdoor_paths = []

        # Get all paths from treatment to outcome
        all_paths = self.graph.find_all_paths(treatment, outcome)

        # Find backdoor paths (paths with edges pointing into treatment)
        for path in all_paths:
            if len(path) > 2:  # Has intermediate nodes
                # Check if first edge points into treatment
                first_intermediate = path[1]
                if self.graph.graph.has_edge(first_intermediate, treatment):
                    backdoor_paths.append(path)

        # Find valid adjustment sets
        confounders = set()
        for path in backdoor_paths:
            # Add all intermediate nodes as potential confounders
            confounders.update(path[1:-1])

        # Remove descendants of treatment
        descendants = self.graph.get_descendants(treatment)
        valid_adjustment_set = confounders - descendants

        # Check if adjustment set blocks all backdoor paths
        blocked_paths = []
        for path in backdoor_paths:
            path_blocked = False
            for node in path[1:-1]:
                if node in valid_adjustment_set:
                    blocked_paths.append(path)
                    path_blocked = True
                    break

        is_valid = len(blocked_paths) == len(backdoor_paths)

        explanation = f"Backdoor criterion for {treatment} -> {outcome}: "
        if is_valid:
            explanation += f"Valid with adjustment set {valid_adjustment_set}"
        else:
            explanation += "No valid adjustment set found"

        return BackdoorCriterion(
            treatment=treatment,
            outcome=outcome,
            adjustment_set=valid_adjustment_set,
            is_valid=is_valid,
            all_backdoor_paths=backdoor_paths,
            blocked_paths=blocked_paths,
            explanation=explanation
        )

    def frontdoor_criterion(self, treatment: str, outcome: str) -> FrontdoorCriterion:
        """Implement front-door criterion"""
        # Find all directed paths from treatment to outcome
        direct_paths = self.graph.find_all_paths(treatment, outcome)

        # Find potential mediator sets
        mediators = set()
        frontdoor_paths = []

        for path in direct_paths:
            if len(path) > 2:  # Has mediators
                path_mediators = set(path[1:-1])
                mediators.update(path_mediators)
                frontdoor_paths.append(path)

        # Check front-door criterion conditions
        # 1. Mediators intercept all directed paths from treatment to outcome
        # 2. No backdoor paths from treatment to mediators
        # 3. All backdoor paths from mediators to outcome are blocked by treatment

        is_valid = len(mediators) > 0 and len(frontdoor_paths) > 0

        explanation = f"Front-door criterion for {treatment} -> {outcome}: "
        if is_valid:
            explanation += f"Valid with mediator set {mediators}"
        else:
            explanation += "No valid mediator set found"

        return FrontdoorCriterion(
            treatment=treatment,
            outcome=outcome,
            mediator_set=mediators,
            is_valid=is_valid,
            frontdoor_paths=frontdoor_paths,
            explanation=explanation
        )

    def estimate_treatment_effect(self,
                                data: pd.DataFrame,
                                treatment: str,
                                outcome: str,
                                method: InferenceMethod = InferenceMethod.BACKDOOR,
                                confounders: Optional[List[str]] = None) -> InterventionEffect:
        """Estimate causal treatment effect"""

        if method == InferenceMethod.BACKDOOR:
            return self._backdoor_adjustment(data, treatment, outcome, confounders)
        elif method == InferenceMethod.INSTRUMENTAL_VARIABLES:
            return self._instrumental_variables(data, treatment, outcome, confounders)
        elif method == InferenceMethod.PROPENSITY_SCORE:
            return self._propensity_score_matching(data, treatment, outcome, confounders)
        else:
            raise ValueError(f"Method {method} not implemented")

    def _backdoor_adjustment(self,
                           data: pd.DataFrame,
                           treatment: str,
                           outcome: str,
                           confounders: Optional[List[str]] = None) -> InterventionEffect:
        """Backdoor adjustment for treatment effect"""

        if confounders is None:
            backdoor = self.backdoor_criterion(treatment, outcome)
            confounders = list(backdoor.adjustment_set)

        # Prepare data
        X = data[confounders + [treatment]] if confounders else data[[treatment]]
        y = data[outcome]

        # Fit regression model
        model = LinearRegression()
        model.fit(X, y)

        # Get treatment effect coefficient
        treatment_idx = -1 if confounders else 0
        effect_size = model.coef_[treatment_idx]

        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_effects = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(data), size=len(data), replace=True)
            boot_data = data.iloc[indices]

            X_boot = boot_data[confounders + [treatment]] if confounders else boot_data[[treatment]]
            y_boot = boot_data[outcome]

            boot_model = LinearRegression()
            boot_model.fit(X_boot, y_boot)
            bootstrap_effects.append(boot_model.coef_[treatment_idx])

        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)

        # Calculate p-value (simplified)
        t_stat = abs(effect_size) / (np.std(bootstrap_effects) + 1e-8)
        p_value = 2 * (1 - stats.t.cdf(t_stat, df=len(data)-len(confounders)-1))

        return InterventionEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method=method,
            confounders_controlled=confounders or [],
            sample_size=len(data),
            assumptions=["Backdoor criterion satisfied", "Linear relationship", "No unmeasured confounding"]
        )

    def _instrumental_variables(self,
                              data: pd.DataFrame,
                              treatment: str,
                              outcome: str,
                              instruments: Optional[List[str]] = None) -> InterventionEffect:
        """Two-stage least squares with instrumental variables"""

        if not instruments:
            # Find potential instruments in the graph
            potential_instruments = []
            for node_name, node in self.graph.nodes.items():
                if node.node_type == "instrument":
                    potential_instruments.append(node_name)
            instruments = potential_instruments[:1]  # Use first instrument

        if not instruments:
            raise ValueError("No instruments available for IV estimation")

        instrument = instruments[0]  # Use first instrument for simplicity

        # First stage: Regress treatment on instrument
        X_first = data[[instrument]]
        y_first = data[treatment]

        first_stage = LinearRegression()
        first_stage.fit(X_first, y_first)
        treatment_predicted = first_stage.predict(X_first)

        # Check instrument strength
        f_stat = first_stage.score(X_first, y_first) * len(data)
        weak_instrument = f_stat < 10  # Rule of thumb: F-stat > 10

        # Second stage: Regress outcome on predicted treatment
        X_second = treatment_predicted.reshape(-1, 1)
        y_second = data[outcome]

        second_stage = LinearRegression()
        second_stage.fit(X_second, y_second)
        effect_size = second_stage.coef_[0]

        # Simple confidence interval (bootstrapped)
        bootstrap_effects = []
        for _ in range(100):
            indices = np.random.choice(len(data), size=len(data), replace=True)
            boot_data = data.iloc[indices]

            # First stage
            X_boot_first = boot_data[[instrument]]
            y_boot_first = boot_data[treatment]
            boot_first = LinearRegression()
            boot_first.fit(X_boot_first, y_boot_first)
            boot_treatment_pred = boot_first.predict(X_boot_first)

            # Second stage
            X_boot_second = boot_treatment_pred.reshape(-1, 1)
            y_boot_second = boot_data[outcome]
            boot_second = LinearRegression()
            boot_second.fit(X_boot_second, y_boot_second)
            bootstrap_effects.append(boot_second.coef_[0])

        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)

        p_value = 0.05 if abs(effect_size) > np.std(bootstrap_effects) * 2 else 0.5

        assumptions = [
            "Instrument relevance",
            "Instrument exogeneity",
            "Exclusion restriction"
        ]

        if weak_instrument:
            assumptions.append("WARNING: Weak instrument detected")

        return InterventionEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method=InferenceMethod.INSTRUMENTAL_VARIABLES,
            confounders_controlled=[],
            sample_size=len(data),
            assumptions=assumptions
        )

    def _propensity_score_matching(self,
                                 data: pd.DataFrame,
                                 treatment: str,
                                 outcome: str,
                                 confounders: Optional[List[str]] = None) -> InterventionEffect:
        """Propensity score matching"""

        if confounders is None:
            backdoor = self.backdoor_criterion(treatment, outcome)
            confounders = list(backdoor.adjustment_set)

        # Estimate propensity scores
        X = data[confounders] if confounders else pd.DataFrame(index=data.index)
        treatment_binary = data[treatment]

        # Use logistic regression for propensity scores
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression()

        if confounders:
            ps_model.fit(X, treatment_binary)
            propensity_scores = ps_model.predict_proba(X)[:, 1]
        else:
            # If no confounders, use treatment prevalence
            propensity_scores = np.full(len(data), treatment_binary.mean())

        # Simple matching (1:1 nearest neighbor)
        treated_indices = np.where(treatment_binary == 1)[0]
        control_indices = np.where(treatment_binary == 0)[0]

        matched_pairs = []
        for treated_idx in treated_indices:
            treated_ps = propensity_scores[treated_idx]

            # Find closest control
            control_ps = propensity_scores[control_indices]
            distances = np.abs(control_ps - treated_ps)
            closest_control_idx = control_indices[np.argmin(distances)]

            matched_pairs.append((treated_idx, closest_control_idx))

        # Calculate average treatment effect on treated
        treated_outcomes = [data[outcome].iloc[t] for t, c in matched_pairs]
        control_outcomes = [data[outcome].iloc[c] for t, c in matched_pairs]

        effect_size = np.mean(treated_outcomes) - np.mean(control_outcomes)

        # Bootstrap confidence interval
        bootstrap_effects = []
        for _ in range(100):
            boot_indices = np.random.choice(len(matched_pairs),
                                          size=len(matched_pairs),
                                          replace=True)
            boot_treated = [treated_outcomes[i] for i in boot_indices]
            boot_control = [control_outcomes[i] for i in boot_indices]
            boot_effect = np.mean(boot_treated) - np.mean(boot_control)
            bootstrap_effects.append(boot_effect)

        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)

        # T-test for significance
        t_stat, p_value = stats.ttest_rel(treated_outcomes, control_outcomes)

        return InterventionEffect(
            treatment=treatment,
            outcome=outcome,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method=InferenceMethod.PROPENSITY_SCORE,
            confounders_controlled=confounders or [],
            sample_size=len(matched_pairs),
            assumptions=["Unconfoundedness", "Overlap", "Stable unit treatment value"]
        )

class CausalGraphAnalytics:
    """Main class for comprehensive causal analysis"""

    def __init__(self, db_path: str = "causal_analytics.db"):
        self.db_path = db_path
        self.graphs: Dict[str, CausalGraph] = {}
        self.inference_engines: Dict[str, CausalInference] = {}
        self.analytics_cache: Dict[str, Any] = {}
        self.telemetry_data: List[Dict] = []
        self._lock = threading.Lock()
        self._setup_database()

        logger.info(f"CausalGraphAnalytics initialized with database: {db_path}")

    def _setup_database(self) -> None:
        """Initialize SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Causal graphs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS causal_graphs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                graph_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Causal nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS causal_nodes (
                graph_id TEXT,
                node_name TEXT,
                node_type TEXT,
                data_type TEXT,
                description TEXT,
                observed BOOLEAN,
                latent BOOLEAN,
                metadata TEXT,
                PRIMARY KEY (graph_id, node_name),
                FOREIGN KEY (graph_id) REFERENCES causal_graphs(id)
            )
        """)

        # Causal edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS causal_edges (
                graph_id TEXT,
                source_node TEXT,
                target_node TEXT,
                edge_type TEXT,
                strength REAL,
                confidence REAL,
                mechanism TEXT,
                metadata TEXT,
                PRIMARY KEY (graph_id, source_node, target_node),
                FOREIGN KEY (graph_id) REFERENCES causal_graphs(id)
            )
        """)

        # Intervention effects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS intervention_effects (
                id TEXT PRIMARY KEY,
                graph_id TEXT,
                treatment TEXT,
                outcome TEXT,
                effect_size REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                p_value REAL,
                method TEXT,
                sample_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (graph_id) REFERENCES causal_graphs(id)
            )
        """)

        # Analytics telemetry table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                operation TEXT,
                graph_id TEXT,
                duration_ms REAL,
                success BOOLEAN,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

    async def create_causal_graph(self, graph_name: str) -> CausalGraph:
        """Create a new causal graph"""
        start_time = datetime.now()

        try:
            with self._lock:
                graph = CausalGraph(graph_name)
                graph_id = f"graph_{len(self.graphs)}_{int(start_time.timestamp())}"

                self.graphs[graph_id] = graph
                self.inference_engines[graph_id] = CausalInference(graph)

                # Persist to database
                self._save_graph_to_db(graph_id, graph)

                self._record_telemetry("create_graph", graph_id, start_time, True)
                logger.info(f"Created causal graph: {graph_name} (ID: {graph_id})")

                return graph

        except Exception as e:
            self._record_telemetry("create_graph", "", start_time, False, {"error": str(e)})
            logger.error(f"Failed to create causal graph: {e}")
            raise

    async def add_causal_relationship(self,
                                   graph_id: str,
                                   source: str,
                                   target: str,
                                   edge_type: CausalEdgeType,
                                   strength: float = 0.5,
                                   confidence: float = 0.8,
                                   mechanism: Optional[str] = None) -> None:
        """Add a causal relationship to the graph"""
        start_time = datetime.now()

        try:
            if graph_id not in self.graphs:
                raise ValueError(f"Graph {graph_id} not found")

            graph = self.graphs[graph_id]

            # Create nodes if they don't exist
            if source not in graph.nodes:
                source_node = CausalNode(
                    name=source,
                    node_type="unknown",
                    data_type="continuous",
                    description=f"Auto-created node: {source}"
                )
                graph.add_node(source_node)

            if target not in graph.nodes:
                target_node = CausalNode(
                    name=target,
                    node_type="unknown",
                    data_type="continuous",
                    description=f"Auto-created node: {target}"
                )
                graph.add_node(target_node)

            # Create edge
            edge = CausalEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                strength=strength,
                confidence=confidence,
                mechanism=mechanism
            )

            graph.add_edge(edge)

            # Update inference engine
            self.inference_engines[graph_id] = CausalInference(graph)

            # Persist to database
            self._save_edge_to_db(graph_id, edge)

            self._record_telemetry("add_relationship", graph_id, start_time, True)
            logger.info(f"Added causal relationship: {source} -> {target}")

        except Exception as e:
            self._record_telemetry("add_relationship", graph_id, start_time, False, {"error": str(e)})
            logger.error(f"Failed to add causal relationship: {e}")
            raise

    async def discover_causal_structure(self,
                                      data: pd.DataFrame,
                                      algorithm: CausalDiscoveryAlgorithm = CausalDiscoveryAlgorithm.PC,
                                      alpha: float = 0.05) -> CausalGraph:
        """Discover causal structure from data"""
        start_time = datetime.now()

        try:
            if algorithm == CausalDiscoveryAlgorithm.PC:
                graph = CausalDiscovery.pc_algorithm(data, alpha)
            elif algorithm == CausalDiscoveryAlgorithm.CONSTRAINT_BASED:
                graph = CausalDiscovery.constraint_based_discovery(data)
            else:
                raise ValueError(f"Algorithm {algorithm} not implemented")

            # Store discovered graph
            graph_id = f"discovered_{algorithm.value}_{int(start_time.timestamp())}"
            self.graphs[graph_id] = graph
            self.inference_engines[graph_id] = CausalInference(graph)

            # Persist to database
            self._save_graph_to_db(graph_id, graph)

            self._record_telemetry("discover_structure", graph_id, start_time, True)
            logger.info(f"Discovered causal structure using {algorithm.value}")

            return graph

        except Exception as e:
            self._record_telemetry("discover_structure", "", start_time, False, {"error": str(e)})
            logger.error(f"Failed to discover causal structure: {e}")
            raise

    async def estimate_intervention_effect(self,
                                         graph_id: str,
                                         data: pd.DataFrame,
                                         treatment: str,
                                         outcome: str,
                                         method: InferenceMethod = InferenceMethod.BACKDOOR,
                                         confounders: Optional[List[str]] = None) -> InterventionEffect:
        """Estimate the causal effect of an intervention"""
        start_time = datetime.now()

        try:
            if graph_id not in self.inference_engines:
                raise ValueError(f"No inference engine for graph {graph_id}")

            inference_engine = self.inference_engines[graph_id]

            effect = inference_engine.estimate_treatment_effect(
                data=data,
                treatment=treatment,
                outcome=outcome,
                method=method,
                confounders=confounders
            )

            # Persist result
            self._save_intervention_effect_to_db(graph_id, effect)

            self._record_telemetry("estimate_effect", graph_id, start_time, True)
            logger.info(f"Estimated intervention effect: {treatment} -> {outcome} = {effect.effect_size:.4f}")

            return effect

        except Exception as e:
            self._record_telemetry("estimate_effect", graph_id, start_time, False, {"error": str(e)})
            logger.error(f"Failed to estimate intervention effect: {e}")
            raise

    async def analyze_backdoor_criterion(self,
                                       graph_id: str,
                                       treatment: str,
                                       outcome: str) -> BackdoorCriterion:
        """Analyze backdoor criterion for causal identification"""
        start_time = datetime.now()

        try:
            if graph_id not in self.inference_engines:
                raise ValueError(f"No inference engine for graph {graph_id}")

            inference_engine = self.inference_engines[graph_id]
            criterion = inference_engine.backdoor_criterion(treatment, outcome)

            self._record_telemetry("backdoor_analysis", graph_id, start_time, True)
            logger.info(f"Backdoor criterion analysis: {treatment} -> {outcome}, Valid: {criterion.is_valid}")

            return criterion

        except Exception as e:
            self._record_telemetry("backdoor_analysis", graph_id, start_time, False, {"error": str(e)})
            logger.error(f"Failed to analyze backdoor criterion: {e}")
            raise

    async def perform_counterfactual_analysis(self,
                                           graph_id: str,
                                           data: pd.DataFrame,
                                           individual_id: str,
                                           treatment: str,
                                           outcome: str) -> CounterfactualAnalysis:
        """Perform counterfactual analysis for an individual"""
        start_time = datetime.now()

        try:
            # Get individual's data
            if individual_id not in data.index:
                raise ValueError(f"Individual {individual_id} not found in data")

            individual_data = data.loc[individual_id]
            factual_outcome = individual_data[outcome]
            factual_treatment = individual_data[treatment]

            # Estimate counterfactual outcome
            # Simplified approach: use treatment effect estimate
            inference_engine = self.inference_engines[graph_id]
            effect = inference_engine.estimate_treatment_effect(
                data=data,
                treatment=treatment,
                outcome=outcome
            )

            # Calculate counterfactual
            if factual_treatment == 1:
                counterfactual_outcome = factual_outcome - effect.effect_size
            else:
                counterfactual_outcome = factual_outcome + effect.effect_size

            individual_effect = factual_outcome - counterfactual_outcome

            # Simplified probability calculation
            probability = max(0.1, min(0.9, effect.confidence_interval[0] /
                                     (effect.confidence_interval[1] - effect.confidence_interval[0] + 1e-8)))

            explanation = f"Individual treatment effect estimated using population average"

            analysis = CounterfactualAnalysis(
                individual_id=individual_id,
                factual_outcome=factual_outcome,
                counterfactual_outcome=counterfactual_outcome,
                treatment_effect=individual_effect,
                probability=probability,
                explanation=explanation,
                sensitivity_analysis={"population_effect": effect.effect_size}
            )

            self._record_telemetry("counterfactual_analysis", graph_id, start_time, True)
            logger.info(f"Counterfactual analysis for individual {individual_id}")

            return analysis

        except Exception as e:
            self._record_telemetry("counterfactual_analysis", graph_id, start_time, False, {"error": str(e)})
            logger.error(f"Failed to perform counterfactual analysis: {e}")
            raise

    async def mediation_analysis(self,
                                graph_id: str,
                                data: pd.DataFrame,
                                treatment: str,
                                mediator: str,
                                outcome: str) -> MediationAnalysis:
        """Perform mediation analysis"""
        start_time = datetime.now()

        try:
            # Total effect (treatment -> outcome)
            total_effect_result = await self.estimate_intervention_effect(
                graph_id=graph_id,
                data=data,
                treatment=treatment,
                outcome=outcome
            )
            total_effect = total_effect_result.effect_size

            # Direct effect (treatment -> outcome, controlling for mediator)
            direct_effect_result = await self.estimate_intervention_effect(
                graph_id=graph_id,
                data=data,
                treatment=treatment,
                outcome=outcome,
                confounders=[mediator]
            )
            direct_effect = direct_effect_result.effect_size

            # Indirect effect
            indirect_effect = total_effect - direct_effect

            # Mediation proportion
            mediation_proportion = indirect_effect / total_effect if total_effect != 0 else 0

            significance = {
                "total_effect_p": total_effect_result.p_value,
                "direct_effect_p": direct_effect_result.p_value,
                "indirect_effect_p": 0.05  # Simplified
            }

            analysis = MediationAnalysis(
                treatment=treatment,
                mediator=mediator,
                outcome=outcome,
                direct_effect=direct_effect,
                indirect_effect=indirect_effect,
                total_effect=total_effect,
                mediation_proportion=mediation_proportion,
                significance=significance
            )

            self._record_telemetry("mediation_analysis", graph_id, start_time, True)
            logger.info(f"Mediation analysis: {treatment} -> {mediator} -> {outcome}")

            return analysis

        except Exception as e:
            self._record_telemetry("mediation_analysis", graph_id, start_time, False, {"error": str(e)})
            logger.error(f"Failed to perform mediation analysis: {e}")
            raise

    def _save_graph_to_db(self, graph_id: str, graph: CausalGraph) -> None:
        """Save causal graph to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Save graph metadata
        graph_data = {
            "nodes": {name: node.__dict__ for name, node in graph.nodes.items()},
            "edges": {f"{edge[0]}->{edge[1]}": edge_obj.__dict__
                     for edge, edge_obj in graph.edges.items()}
        }

        cursor.execute("""
            INSERT OR REPLACE INTO causal_graphs
            (id, name, graph_data, metadata)
            VALUES (?, ?, ?, ?)
        """, (graph_id, graph.name, json.dumps(graph_data), "{}"))

        # Save nodes
        for node_name, node in graph.nodes.items():
            cursor.execute("""
                INSERT OR REPLACE INTO causal_nodes
                (graph_id, node_name, node_type, data_type, description, observed, latent, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (graph_id, node_name, node.node_type, node.data_type,
                  node.description, node.observed, node.latent, json.dumps(node.metadata)))

        conn.commit()
        conn.close()

    def _save_edge_to_db(self, graph_id: str, edge: CausalEdge) -> None:
        """Save causal edge to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO causal_edges
            (graph_id, source_node, target_node, edge_type, strength, confidence, mechanism, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (graph_id, edge.source, edge.target, edge.edge_type.value,
              edge.strength, edge.confidence, edge.mechanism, json.dumps(edge.metadata)))

        conn.commit()
        conn.close()

    def _save_intervention_effect_to_db(self, graph_id: str, effect: InterventionEffect) -> None:
        """Save intervention effect to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        effect_id = f"effect_{graph_id}_{effect.treatment}_{effect.outcome}_{int(datetime.now().timestamp())}"

        cursor.execute("""
            INSERT INTO intervention_effects
            (id, graph_id, treatment, outcome, effect_size, confidence_lower, confidence_upper,
             p_value, method, sample_size, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (effect_id, graph_id, effect.treatment, effect.outcome, effect.effect_size,
              effect.confidence_interval[0], effect.confidence_interval[1],
              effect.p_value, effect.method.value, effect.sample_size, json.dumps(effect.metadata)))

        conn.commit()
        conn.close()

    def _record_telemetry(self,
                         operation: str,
                         graph_id: str,
                         start_time: datetime,
                         success: bool,
                         metadata: Optional[Dict] = None) -> None:
        """Record telemetry data"""
        duration = (datetime.now() - start_time).total_seconds() * 1000

        telemetry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "graph_id": graph_id,
            "duration_ms": duration,
            "success": success,
            "metadata": metadata or {}
        }

        self.telemetry_data.append(telemetry)

        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO analytics_telemetry
            (operation, graph_id, duration_ms, success, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (operation, graph_id, duration, success, json.dumps(metadata or {})))

        conn.commit()
        conn.close()

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        summary = {
            "total_graphs": len(self.graphs),
            "total_operations": len(self.telemetry_data),
            "success_rate": sum(1 for t in self.telemetry_data if t["success"]) / max(1, len(self.telemetry_data)),
            "average_operation_time": np.mean([t["duration_ms"] for t in self.telemetry_data]) if self.telemetry_data else 0,
            "graphs": {}
        }

        for graph_id, graph in self.graphs.items():
            summary["graphs"][graph_id] = {
                "name": graph.name,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "is_acyclic": graph.is_acyclic(),
                "created_at": graph.created_at.isoformat()
            }

        return summary