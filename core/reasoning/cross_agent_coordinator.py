"""
Cross-Agent Reasoning Coordinator - GPT-5 Priority 2
Prevents logical conflicts and ensures reasoning consistency across all agents.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import hashlib

from ..observability.telemetry_engine import get_telemetry, LogLevel

class ReasoningConflictType(Enum):
    LOGICAL_CONTRADICTION = "logical_contradiction"
    RESOURCE_CONFLICT = "resource_conflict"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    DEPENDENCY_VIOLATION = "dependency_violation"
    GOAL_MISALIGNMENT = "goal_misalignment"

class ConflictSeverity(Enum):
    CRITICAL = "critical"    # Blocks all progress
    HIGH = "high"           # Blocks specific agents
    MEDIUM = "medium"       # Degrades performance
    LOW = "low"            # Minor inconsistency

@dataclass
class ReasoningContext:
    """Agent reasoning context and state"""
    agent_id: str
    operation: str
    reasoning_chain: List[Dict[str, Any]]
    assumptions: List[str]
    dependencies: List[str]
    timestamp: datetime
    confidence: float
    expected_outcomes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'operation': self.operation,
            'reasoning_chain': self.reasoning_chain,
            'assumptions': self.assumptions,
            'dependencies': self.dependencies,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'expected_outcomes': self.expected_outcomes
        }

@dataclass
class ReasoningConflict:
    """Detected reasoning conflict between agents"""
    conflict_id: str
    conflict_type: ReasoningConflictType
    severity: ConflictSeverity
    involved_agents: List[str]
    description: str
    conflicting_elements: Dict[str, Any]
    detection_timestamp: datetime
    resolution_strategy: Optional[str] = None
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'conflict_id': self.conflict_id,
            'conflict_type': self.conflict_type.value,
            'severity': self.severity.value,
            'involved_agents': self.involved_agents,
            'description': self.description,
            'conflicting_elements': self.conflicting_elements,
            'detection_timestamp': self.detection_timestamp.isoformat(),
            'resolution_strategy': self.resolution_strategy,
            'resolved': self.resolved
        }

@dataclass
class ConsensusDecision:
    """Multi-agent consensus decision"""
    decision_id: str
    topic: str
    participating_agents: List[str]
    individual_inputs: Dict[str, ReasoningContext]
    consensus_reached: bool
    final_decision: Optional[Dict[str, Any]]
    confidence_level: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'topic': self.topic,
            'participating_agents': self.participating_agents,
            'individual_inputs': {k: v.to_dict() for k, v in self.individual_inputs.items()},
            'consensus_reached': self.consensus_reached,
            'final_decision': self.final_decision,
            'confidence_level': self.confidence_level,
            'timestamp': self.timestamp.isoformat()
        }

class CrossAgentCoordinator:
    """
    Cross-Agent Reasoning Coordinator

    Manages reasoning consistency and conflict resolution across multiple agents:
    - Detects logical conflicts before they cause system issues
    - Validates reasoning chains for consistency
    - Facilitates consensus-building for major decisions
    - Maintains shared knowledge state across agents
    """

    def __init__(self, telemetry=None):
        self.telemetry = telemetry or get_telemetry()
        self.telemetry.register_component('cross_agent_coordinator')

        # Agent tracking
        self.active_contexts = {}  # agent_id -> ReasoningContext
        self.agent_knowledge_bases = defaultdict(dict)  # agent_id -> knowledge
        self.shared_knowledge = {}  # Cross-agent shared knowledge

        # Conflict management
        self.active_conflicts = {}  # conflict_id -> ReasoningConflict
        self.conflict_history = deque(maxlen=1000)
        self.resolution_strategies = {}

        # Consensus management
        self.active_consensus = {}  # decision_id -> ConsensusDecision
        self.consensus_history = deque(maxlen=500)

        # Configuration
        self.conflict_detection_enabled = True
        self.consensus_threshold = 0.75  # 75% agreement required
        self.max_reasoning_depth = 10

        # Threading
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread = None

        # Performance tracking
        self.coordination_metrics = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'consensus_decisions': 0,
            'reasoning_validations': 0
        }

        self._setup_conflict_detectors()
        self._setup_resolution_strategies()

    async def start(self):
        """Start cross-agent coordination"""
        self.telemetry.log_info("Starting cross-agent reasoning coordinator", 'cross_agent_coordinator')

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        self.telemetry.log_info("Cross-agent coordinator started", 'cross_agent_coordinator')

    async def stop(self):
        """Stop coordination"""
        self.telemetry.log_info("Stopping cross-agent coordinator", 'cross_agent_coordinator')

        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)

        # Resolve any pending conflicts
        await self._emergency_conflict_resolution()

        self.telemetry.log_info("Cross-agent coordinator stopped", 'cross_agent_coordinator')

    def register_agent_context(self, context: ReasoningContext) -> bool:
        """Register agent reasoning context and check for conflicts"""
        with self._lock:
            self.active_contexts[context.agent_id] = context

        # Validate reasoning chain
        validation_result = self._validate_reasoning_chain(context)
        if not validation_result['valid']:
            self.telemetry.log_warning(
                f"Invalid reasoning chain from {context.agent_id}",
                'cross_agent_coordinator',
                {'validation_errors': validation_result['errors']}
            )
            return False

        # Check for conflicts with other agents
        conflicts = self._detect_conflicts(context)
        if conflicts:
            for conflict in conflicts:
                self._handle_conflict(conflict)

        # Update agent knowledge base
        self._update_agent_knowledge(context)

        self.coordination_metrics['reasoning_validations'] += 1

        return len(conflicts) == 0

    def request_consensus(self, topic: str, participating_agents: List[str],
                         timeout_seconds: int = 300) -> Optional[ConsensusDecision]:
        """Request consensus decision from multiple agents"""
        decision_id = str(uuid.uuid4())

        decision = ConsensusDecision(
            decision_id=decision_id,
            topic=topic,
            participating_agents=participating_agents,
            individual_inputs={},
            consensus_reached=False,
            final_decision=None,
            confidence_level=0.0,
            timestamp=datetime.now(timezone.utc)
        )

        with self._lock:
            self.active_consensus[decision_id] = decision

        self.telemetry.log_info(
            f"Consensus requested: {topic}",
            'cross_agent_coordinator',
            {
                'decision_id': decision_id,
                'participating_agents': participating_agents,
                'timeout_seconds': timeout_seconds
            }
        )

        # Note: In a real implementation, this would be async
        # and would actually request input from each agent
        return decision

    def submit_consensus_input(self, decision_id: str, agent_id: str,
                             reasoning_context: ReasoningContext) -> bool:
        """Submit agent input for consensus decision"""
        with self._lock:
            if decision_id not in self.active_consensus:
                return False

            decision = self.active_consensus[decision_id]
            if agent_id not in decision.participating_agents:
                return False

            decision.individual_inputs[agent_id] = reasoning_context

        # Check if we have all inputs
        self._evaluate_consensus(decision_id)

        return True

    def get_shared_knowledge(self, knowledge_type: str = None) -> Dict[str, Any]:
        """Get shared knowledge accessible to all agents"""
        with self._lock:
            if knowledge_type:
                return self.shared_knowledge.get(knowledge_type, {})
            return dict(self.shared_knowledge)

    def update_shared_knowledge(self, knowledge_type: str, data: Dict[str, Any],
                              source_agent: str) -> bool:
        """Update shared knowledge from an agent"""
        # Validate knowledge consistency
        if not self._validate_knowledge_consistency(knowledge_type, data):
            self.telemetry.log_warning(
                f"Knowledge update rejected due to inconsistency",
                'cross_agent_coordinator',
                {
                    'knowledge_type': knowledge_type,
                    'source_agent': source_agent,
                    'data_keys': list(data.keys())
                }
            )
            return False

        with self._lock:
            if knowledge_type not in self.shared_knowledge:
                self.shared_knowledge[knowledge_type] = {}

            # Merge with conflict resolution
            self.shared_knowledge[knowledge_type].update(data)

        self.telemetry.log_info(
            f"Shared knowledge updated: {knowledge_type}",
            'cross_agent_coordinator',
            {'source_agent': source_agent, 'keys_updated': len(data)}
        )

        return True

    def get_active_conflicts(self, severity: ConflictSeverity = None) -> List[ReasoningConflict]:
        """Get active conflicts, optionally filtered by severity"""
        with self._lock:
            conflicts = list(self.active_conflicts.values())

        if severity:
            conflicts = [c for c in conflicts if c.severity == severity]

        return sorted(conflicts, key=lambda x: (x.severity.value, x.detection_timestamp))

    def resolve_conflict(self, conflict_id: str, resolution_strategy: str) -> bool:
        """Manually resolve a conflict"""
        with self._lock:
            if conflict_id not in self.active_conflicts:
                return False

            conflict = self.active_conflicts[conflict_id]

        success = self._apply_resolution_strategy(conflict, resolution_strategy)

        if success:
            conflict.resolved = True
            conflict.resolution_strategy = resolution_strategy

            with self._lock:
                del self.active_conflicts[conflict_id]
                self.conflict_history.append(conflict)

            self.coordination_metrics['conflicts_resolved'] += 1

            self.telemetry.log_info(
                f"Conflict resolved: {conflict_id}",
                'cross_agent_coordinator',
                {
                    'conflict_type': conflict.conflict_type.value,
                    'resolution_strategy': resolution_strategy,
                    'involved_agents': conflict.involved_agents
                }
            )

        return success

    def get_coordination_status(self) -> Dict[str, Any]:
        """Get coordination system status"""
        with self._lock:
            active_conflicts_count = len(self.active_conflicts)
            critical_conflicts = len([
                c for c in self.active_conflicts.values()
                if c.severity == ConflictSeverity.CRITICAL
            ])
            active_consensus_count = len(self.active_consensus)
            active_agents_count = len(self.active_contexts)

        return {
            'status': 'healthy' if critical_conflicts == 0 else 'degraded',
            'active_agents': active_agents_count,
            'active_conflicts': active_conflicts_count,
            'critical_conflicts': critical_conflicts,
            'active_consensus_decisions': active_consensus_count,
            'coordination_metrics': dict(self.coordination_metrics),
            'shared_knowledge_types': list(self.shared_knowledge.keys())
        }

    # Internal methods

    def _setup_conflict_detectors(self):
        """Setup conflict detection strategies"""
        self.conflict_detectors = {
            ReasoningConflictType.LOGICAL_CONTRADICTION: self._detect_logical_contradictions,
            ReasoningConflictType.RESOURCE_CONFLICT: self._detect_resource_conflicts,
            ReasoningConflictType.TEMPORAL_INCONSISTENCY: self._detect_temporal_inconsistencies,
            ReasoningConflictType.DEPENDENCY_VIOLATION: self._detect_dependency_violations,
            ReasoningConflictType.GOAL_MISALIGNMENT: self._detect_goal_misalignments
        }

    def _setup_resolution_strategies(self):
        """Setup conflict resolution strategies"""
        self.resolution_strategies = {
            'priority_override': self._resolve_by_priority,
            'consensus_vote': self._resolve_by_consensus,
            'temporal_ordering': self._resolve_by_temporal_order,
            'resource_allocation': self._resolve_by_resource_allocation,
            'escalation': self._escalate_to_human
        }

    def _validate_reasoning_chain(self, context: ReasoningContext) -> Dict[str, Any]:
        """Validate logical consistency of reasoning chain"""
        errors = []

        # Check chain depth
        if len(context.reasoning_chain) > self.max_reasoning_depth:
            errors.append("Reasoning chain too deep")

        # Check for circular reasoning
        reasoning_hashes = []
        for step in context.reasoning_chain:
            step_hash = hashlib.md5(json.dumps(step, sort_keys=True).encode()).hexdigest()
            if step_hash in reasoning_hashes:
                errors.append("Circular reasoning detected")
                break
            reasoning_hashes.append(step_hash)

        # Check for contradictions within chain
        for i, step in enumerate(context.reasoning_chain):
            for j, other_step in enumerate(context.reasoning_chain[i+1:], i+1):
                if self._steps_contradict(step, other_step):
                    errors.append(f"Contradiction between steps {i} and {j}")

        # Validate confidence level
        if context.confidence < 0 or context.confidence > 1:
            errors.append("Invalid confidence level")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'chain_length': len(context.reasoning_chain)
        }

    def _detect_conflicts(self, context: ReasoningContext) -> List[ReasoningConflict]:
        """Detect conflicts with other active contexts"""
        conflicts = []

        with self._lock:
            other_contexts = {
                agent_id: ctx for agent_id, ctx in self.active_contexts.items()
                if agent_id != context.agent_id
            }

        for conflict_type, detector in self.conflict_detectors.items():
            conflict = detector(context, other_contexts)
            if conflict:
                conflicts.append(conflict)

        return conflicts

    def _detect_logical_contradictions(self, context: ReasoningContext,
                                     other_contexts: Dict[str, ReasoningContext]) -> Optional[ReasoningConflict]:
        """Detect logical contradictions between agent reasoning"""
        for other_agent_id, other_context in other_contexts.items():
            # Check if assumptions contradict
            for assumption in context.assumptions:
                for other_assumption in other_context.assumptions:
                    if self._assumptions_contradict(assumption, other_assumption):
                        return ReasoningConflict(
                            conflict_id=str(uuid.uuid4()),
                            conflict_type=ReasoningConflictType.LOGICAL_CONTRADICTION,
                            severity=ConflictSeverity.HIGH,
                            involved_agents=[context.agent_id, other_agent_id],
                            description=f"Contradictory assumptions: '{assumption}' vs '{other_assumption}'",
                            conflicting_elements={
                                'assumption1': assumption,
                                'assumption2': other_assumption,
                                'agent1': context.agent_id,
                                'agent2': other_agent_id
                            },
                            detection_timestamp=datetime.now(timezone.utc)
                        )

            # Check if expected outcomes contradict
            for outcome in context.expected_outcomes:
                for other_outcome in other_context.expected_outcomes:
                    if self._outcomes_contradict(outcome, other_outcome):
                        return ReasoningConflict(
                            conflict_id=str(uuid.uuid4()),
                            conflict_type=ReasoningConflictType.LOGICAL_CONTRADICTION,
                            severity=ConflictSeverity.MEDIUM,
                            involved_agents=[context.agent_id, other_agent_id],
                            description=f"Contradictory outcomes: '{outcome}' vs '{other_outcome}'",
                            conflicting_elements={
                                'outcome1': outcome,
                                'outcome2': other_outcome,
                                'agent1': context.agent_id,
                                'agent2': other_agent_id
                            },
                            detection_timestamp=datetime.now(timezone.utc)
                        )

        return None

    def _detect_resource_conflicts(self, context: ReasoningContext,
                                 other_contexts: Dict[str, ReasoningContext]) -> Optional[ReasoningConflict]:
        """Detect resource conflicts between agents"""
        # Extract resource references from reasoning chain
        context_resources = self._extract_resource_references(context)

        for other_agent_id, other_context in other_contexts.items():
            other_resources = self._extract_resource_references(other_context)

            # Check for write-write conflicts
            write_conflicts = context_resources.get('write', set()) & other_resources.get('write', set())
            if write_conflicts:
                return ReasoningConflict(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ReasoningConflictType.RESOURCE_CONFLICT,
                    severity=ConflictSeverity.CRITICAL,
                    involved_agents=[context.agent_id, other_agent_id],
                    description=f"Write-write conflict on resources: {write_conflicts}",
                    conflicting_elements={
                        'conflicting_resources': list(write_conflicts),
                        'agent1': context.agent_id,
                        'agent2': other_agent_id
                    },
                    detection_timestamp=datetime.now(timezone.utc)
                )

        return None

    def _detect_temporal_inconsistencies(self, context: ReasoningContext,
                                       other_contexts: Dict[str, ReasoningContext]) -> Optional[ReasoningConflict]:
        """Detect temporal inconsistencies between agent plans"""
        # Check for temporal ordering conflicts in dependencies
        for dependency in context.dependencies:
            for other_agent_id, other_context in other_contexts.items():
                if dependency in other_context.dependencies:
                    # Check timestamps to detect potential race conditions
                    time_diff = abs((context.timestamp - other_context.timestamp).total_seconds())
                    if time_diff < 1:  # Less than 1 second apart
                        return ReasoningConflict(
                            conflict_id=str(uuid.uuid4()),
                            conflict_type=ReasoningConflictType.TEMPORAL_INCONSISTENCY,
                            severity=ConflictSeverity.MEDIUM,
                            involved_agents=[context.agent_id, other_agent_id],
                            description=f"Potential race condition on dependency: {dependency}",
                            conflicting_elements={
                                'dependency': dependency,
                                'time_diff_seconds': time_diff,
                                'agent1_timestamp': context.timestamp.isoformat(),
                                'agent2_timestamp': other_context.timestamp.isoformat()
                            },
                            detection_timestamp=datetime.now(timezone.utc)
                        )

        return None

    def _detect_dependency_violations(self, context: ReasoningContext,
                                    other_contexts: Dict[str, ReasoningContext]) -> Optional[ReasoningConflict]:
        """Detect dependency violations between agents"""
        # Check if this agent's operation violates dependencies of other agents
        for other_agent_id, other_context in other_contexts.items():
            for dependency in other_context.dependencies:
                if self._operation_violates_dependency(context.operation, dependency):
                    return ReasoningConflict(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type=ReasoningConflictType.DEPENDENCY_VIOLATION,
                        severity=ConflictSeverity.HIGH,
                        involved_agents=[context.agent_id, other_agent_id],
                        description=f"Operation '{context.operation}' violates dependency '{dependency}'",
                        conflicting_elements={
                            'operation': context.operation,
                            'violated_dependency': dependency,
                            'violating_agent': context.agent_id,
                            'dependent_agent': other_agent_id
                        },
                        detection_timestamp=datetime.now(timezone.utc)
                    )

        return None

    def _detect_goal_misalignments(self, context: ReasoningContext,
                                 other_contexts: Dict[str, ReasoningContext]) -> Optional[ReasoningConflict]:
        """Detect goal misalignments between agents"""
        # Extract goals from expected outcomes
        context_goals = set(context.expected_outcomes)

        for other_agent_id, other_context in other_contexts.items():
            other_goals = set(other_context.expected_outcomes)

            # Check for directly opposing goals
            for goal in context_goals:
                for other_goal in other_goals:
                    if self._goals_oppose(goal, other_goal):
                        return ReasoningConflict(
                            conflict_id=str(uuid.uuid4()),
                            conflict_type=ReasoningConflictType.GOAL_MISALIGNMENT,
                            severity=ConflictSeverity.MEDIUM,
                            involved_agents=[context.agent_id, other_agent_id],
                            description=f"Opposing goals: '{goal}' vs '{other_goal}'",
                            conflicting_elements={
                                'goal1': goal,
                                'goal2': other_goal,
                                'agent1': context.agent_id,
                                'agent2': other_agent_id
                            },
                            detection_timestamp=datetime.now(timezone.utc)
                        )

        return None

    def _handle_conflict(self, conflict: ReasoningConflict):
        """Handle detected conflict"""
        with self._lock:
            self.active_conflicts[conflict.conflict_id] = conflict

        self.coordination_metrics['conflicts_detected'] += 1

        self.telemetry.log_warning(
            f"Reasoning conflict detected: {conflict.conflict_type.value}",
            'cross_agent_coordinator',
            {
                'conflict_id': conflict.conflict_id,
                'severity': conflict.severity.value,
                'involved_agents': conflict.involved_agents,
                'description': conflict.description
            }
        )

        # Auto-resolve if possible
        if conflict.severity in [ConflictSeverity.LOW, ConflictSeverity.MEDIUM]:
            self._auto_resolve_conflict(conflict)

    def _auto_resolve_conflict(self, conflict: ReasoningConflict):
        """Attempt automatic conflict resolution"""
        # Choose resolution strategy based on conflict type
        strategy_map = {
            ReasoningConflictType.LOGICAL_CONTRADICTION: 'consensus_vote',
            ReasoningConflictType.RESOURCE_CONFLICT: 'resource_allocation',
            ReasoningConflictType.TEMPORAL_INCONSISTENCY: 'temporal_ordering',
            ReasoningConflictType.DEPENDENCY_VIOLATION: 'priority_override',
            ReasoningConflictType.GOAL_MISALIGNMENT: 'consensus_vote'
        }

        strategy = strategy_map.get(conflict.conflict_type, 'escalation')
        self.resolve_conflict(conflict.conflict_id, strategy)

    def _apply_resolution_strategy(self, conflict: ReasoningConflict, strategy: str) -> bool:
        """Apply specific resolution strategy"""
        resolver = self.resolution_strategies.get(strategy)
        if not resolver:
            return False

        return resolver(conflict)

    # Conflict resolution strategies

    def _resolve_by_priority(self, conflict: ReasoningConflict) -> bool:
        """Resolve by agent priority"""
        # Simple implementation: first agent wins
        # In real system, would use actual agent priorities
        return True

    def _resolve_by_consensus(self, conflict: ReasoningConflict) -> bool:
        """Resolve by requesting consensus vote"""
        # In real system, would initiate consensus process
        return True

    def _resolve_by_temporal_order(self, conflict: ReasoningConflict) -> bool:
        """Resolve by temporal ordering (first come, first served)"""
        return True

    def _resolve_by_resource_allocation(self, conflict: ReasoningConflict) -> bool:
        """Resolve by resource allocation strategy"""
        return True

    def _escalate_to_human(self, conflict: ReasoningConflict) -> bool:
        """Escalate conflict to human operator"""
        self.telemetry.log_critical(
            f"Conflict escalated to human: {conflict.conflict_id}",
            'cross_agent_coordinator',
            conflict.to_dict()
        )
        return False

    # Helper methods

    def _steps_contradict(self, step1: Dict[str, Any], step2: Dict[str, Any]) -> bool:
        """Check if two reasoning steps contradict each other"""
        # Simple heuristic implementation
        # In real system, would use more sophisticated logic
        return False

    def _assumptions_contradict(self, assumption1: str, assumption2: str) -> bool:
        """Check if two assumptions contradict"""
        # Simple keyword-based contradiction detection
        negation_pairs = [
            ('possible', 'impossible'),
            ('exists', 'does not exist'),
            ('true', 'false'),
            ('enabled', 'disabled'),
            ('available', 'unavailable')
        ]

        a1_lower = assumption1.lower()
        a2_lower = assumption2.lower()

        for pos, neg in negation_pairs:
            if pos in a1_lower and neg in a2_lower:
                return True
            if neg in a1_lower and pos in a2_lower:
                return True

        return False

    def _outcomes_contradict(self, outcome1: str, outcome2: str) -> bool:
        """Check if two expected outcomes contradict"""
        return self._assumptions_contradict(outcome1, outcome2)

    def _goals_oppose(self, goal1: str, goal2: str) -> bool:
        """Check if two goals directly oppose each other"""
        return self._assumptions_contradict(goal1, goal2)

    def _extract_resource_references(self, context: ReasoningContext) -> Dict[str, Set[str]]:
        """Extract resource references from reasoning context"""
        resources = {'read': set(), 'write': set()}

        # Simple extraction from operation and reasoning chain
        # In real system, would parse more sophisticated resource references
        operation_lower = context.operation.lower()

        if 'file' in operation_lower or 'write' in operation_lower:
            # Extract potential file references
            words = operation_lower.split()
            for word in words:
                if '.' in word or '/' in word:
                    resources['write'].add(word)

        if 'read' in operation_lower or 'get' in operation_lower:
            words = operation_lower.split()
            for word in words:
                if '.' in word or '/' in word:
                    resources['read'].add(word)

        return resources

    def _operation_violates_dependency(self, operation: str, dependency: str) -> bool:
        """Check if operation violates a dependency"""
        # Simple heuristic implementation
        operation_lower = operation.lower()
        dependency_lower = dependency.lower()

        # Check for operations that might violate dependencies
        if 'delete' in operation_lower and any(word in dependency_lower for word in ['require', 'need', 'depend']):
            return True

        return False

    def _validate_knowledge_consistency(self, knowledge_type: str, data: Dict[str, Any]) -> bool:
        """Validate that knowledge update is consistent with existing knowledge"""
        with self._lock:
            existing = self.shared_knowledge.get(knowledge_type, {})

        # Check for direct contradictions
        for key, value in data.items():
            if key in existing and existing[key] != value:
                # Allow updates if new value is more recent or has higher confidence
                if isinstance(value, dict) and 'timestamp' in value:
                    if isinstance(existing[key], dict) and 'timestamp' in existing[key]:
                        new_time = datetime.fromisoformat(value['timestamp'])
                        old_time = datetime.fromisoformat(existing[key]['timestamp'])
                        if new_time <= old_time:
                            return False
                else:
                    return False

        return True

    def _update_agent_knowledge(self, context: ReasoningContext):
        """Update agent-specific knowledge base"""
        with self._lock:
            agent_kb = self.agent_knowledge_bases[context.agent_id]

            # Store reasoning patterns
            agent_kb['last_reasoning'] = context.reasoning_chain
            agent_kb['last_operation'] = context.operation
            agent_kb['last_update'] = context.timestamp.isoformat()

            # Update operation history
            if 'operation_history' not in agent_kb:
                agent_kb['operation_history'] = deque(maxlen=100)
            agent_kb['operation_history'].append({
                'operation': context.operation,
                'timestamp': context.timestamp.isoformat(),
                'confidence': context.confidence
            })

    def _evaluate_consensus(self, decision_id: str):
        """Evaluate consensus for a decision"""
        with self._lock:
            decision = self.active_consensus.get(decision_id)
            if not decision:
                return

            # Check if we have all required inputs
            if len(decision.individual_inputs) < len(decision.participating_agents):
                return

            # Calculate consensus
            confidence_scores = [ctx.confidence for ctx in decision.individual_inputs.values()]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)

            # Simple consensus: average confidence above threshold
            if avg_confidence >= self.consensus_threshold:
                decision.consensus_reached = True
                decision.confidence_level = avg_confidence
                decision.final_decision = self._merge_consensus_inputs(decision)

                # Move to history
                self.consensus_history.append(decision)
                del self.active_consensus[decision_id]

                self.coordination_metrics['consensus_decisions'] += 1

                self.telemetry.log_info(
                    f"Consensus reached for: {decision.topic}",
                    'cross_agent_coordinator',
                    {
                        'decision_id': decision_id,
                        'confidence_level': avg_confidence,
                        'participating_agents': len(decision.participating_agents)
                    }
                )

    def _merge_consensus_inputs(self, decision: ConsensusDecision) -> Dict[str, Any]:
        """Merge individual agent inputs into consensus decision"""
        # Simple implementation: combine all reasoning chains
        merged = {
            'topic': decision.topic,
            'reasoning_chains': {
                agent_id: ctx.reasoning_chain
                for agent_id, ctx in decision.individual_inputs.items()
            },
            'combined_assumptions': [],
            'combined_outcomes': []
        }

        # Merge non-conflicting assumptions and outcomes
        for ctx in decision.individual_inputs.values():
            merged['combined_assumptions'].extend(ctx.assumptions)
            merged['combined_outcomes'].extend(ctx.expected_outcomes)

        # Remove duplicates
        merged['combined_assumptions'] = list(set(merged['combined_assumptions']))
        merged['combined_outcomes'] = list(set(merged['combined_outcomes']))

        return merged

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                # Check for stale conflicts
                self._cleanup_stale_conflicts()

                # Check for timeout consensus decisions
                self._timeout_stale_consensus()

                # Update metrics
                self._update_coordination_metrics()

            except Exception as e:
                self.telemetry.log_error(
                    f"Error in coordination monitoring loop: {e}",
                    'cross_agent_coordinator'
                )

            time.sleep(10)  # Monitor every 10 seconds

    def _cleanup_stale_conflicts(self):
        """Clean up conflicts that are too old"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - 3600  # 1 hour

        with self._lock:
            stale_conflicts = [
                conflict_id for conflict_id, conflict in self.active_conflicts.items()
                if conflict.detection_timestamp.timestamp() < cutoff_time
            ]

            for conflict_id in stale_conflicts:
                conflict = self.active_conflicts[conflict_id]
                conflict.resolved = True
                conflict.resolution_strategy = 'timeout'

                self.conflict_history.append(conflict)
                del self.active_conflicts[conflict_id]

                self.telemetry.log_warning(
                    f"Conflict timed out and auto-resolved: {conflict_id}",
                    'cross_agent_coordinator'
                )

    def _timeout_stale_consensus(self):
        """Timeout stale consensus decisions"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - 600  # 10 minutes

        with self._lock:
            stale_decisions = [
                decision_id for decision_id, decision in self.active_consensus.items()
                if decision.timestamp.timestamp() < cutoff_time
            ]

            for decision_id in stale_decisions:
                decision = self.active_consensus[decision_id]
                decision.consensus_reached = False
                decision.final_decision = {'status': 'timeout'}

                self.consensus_history.append(decision)
                del self.active_consensus[decision_id]

                self.telemetry.log_warning(
                    f"Consensus decision timed out: {decision_id}",
                    'cross_agent_coordinator'
                )

    def _update_coordination_metrics(self):
        """Update coordination performance metrics"""
        with self._lock:
            active_conflicts = len(self.active_conflicts)
            critical_conflicts = len([
                c for c in self.active_conflicts.values()
                if c.severity == ConflictSeverity.CRITICAL
            ])

        self.telemetry.set_gauge(
            'coordination_active_conflicts',
            active_conflicts,
            component='cross_agent_coordinator'
        )

        self.telemetry.set_gauge(
            'coordination_critical_conflicts',
            critical_conflicts,
            component='cross_agent_coordinator'
        )

        # Emit coordination health score
        health_score = 100 if critical_conflicts == 0 else max(0, 100 - (critical_conflicts * 25))
        self.telemetry.set_gauge(
            'coordination_health_score',
            health_score,
            component='cross_agent_coordinator'
        )

    async def _emergency_conflict_resolution(self):
        """Emergency resolution of all conflicts on shutdown"""
        with self._lock:
            critical_conflicts = [
                conflict for conflict in self.active_conflicts.values()
                if conflict.severity == ConflictSeverity.CRITICAL
            ]

        for conflict in critical_conflicts:
            self.resolve_conflict(conflict.conflict_id, 'escalation')

        self.telemetry.log_info(
            f"Emergency resolution completed for {len(critical_conflicts)} critical conflicts",
            'cross_agent_coordinator'
        )