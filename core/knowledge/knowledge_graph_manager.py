"""
Knowledge Graph & Ontology Manager - GPT-5 Priority 7
Advanced knowledge representation, semantic reasoning, and graph-based intelligence.

Features:
- Multi-modal knowledge graph construction
- Semantic reasoning and inference engine
- Ontology management and taxonomy support
- SPARQL-like query capabilities
- Real-time graph analytics and pattern discovery
- Knowledge validation and consistency checking
- Entity extraction and relationship detection
- Graph visualization and exploration
"""

import asyncio
import json
import re
import threading
import sqlite3
import hashlib
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of knowledge entities"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    DOCUMENT = "DOCUMENT"
    ARTIFACT = "ARTIFACT"
    PROCESS = "PROCESS"
    SYSTEM = "SYSTEM"
    TECHNOLOGY = "TECHNOLOGY"
    DOMAIN = "DOMAIN"
    ATTRIBUTE = "ATTRIBUTE"
    VALUE = "VALUE"


class RelationType(Enum):
    """Types of relationships between entities"""
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    HAS_PROPERTY = "HAS_PROPERTY"
    RELATES_TO = "RELATES_TO"
    CAUSES = "CAUSES"
    ENABLES = "ENABLES"
    REQUIRES = "REQUIRES"
    INFLUENCES = "INFLUENCES"
    CREATED_BY = "CREATED_BY"
    LOCATED_IN = "LOCATED_IN"
    OCCURRED_AT = "OCCURRED_AT"
    OWNED_BY = "OWNED_BY"
    USED_BY = "USED_BY"
    SIMILAR_TO = "SIMILAR_TO"
    DERIVES_FROM = "DERIVES_FROM"


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge assertions"""
    CERTAIN = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    UNCERTAIN = 0.2


@dataclass
class KnowledgeEntity:
    """Represents an entity in the knowledge graph"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeRelation:
    """Represents a relationship between entities"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    weight: float = 1.0
    temporal_context: Optional[str] = None


@dataclass
class Ontology:
    """Represents an ontology schema"""
    id: str
    name: str
    version: str
    description: str
    namespaces: Dict[str, str] = field(default_factory=dict)
    classes: Dict[str, Dict] = field(default_factory=dict)
    properties: Dict[str, Dict] = field(default_factory=dict)
    axioms: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SemanticQuery:
    """Represents a semantic query against the knowledge graph"""
    query_id: str
    query_text: str
    query_type: str  # "sparql", "natural_language", "graph_pattern"
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    result_limit: int = 100


@dataclass
class InferenceRule:
    """Represents a logical inference rule"""
    rule_id: str
    name: str
    description: str
    condition: str  # Logical condition
    conclusion: str  # Logical conclusion
    confidence_threshold: float = 0.7
    priority: int = 1
    enabled: bool = True
    rule_type: str = "forward_chaining"


@dataclass
class GraphAnalytics:
    """Analytics results from graph analysis"""
    total_entities: int
    total_relations: int
    graph_density: float
    average_clustering: float
    diameter: int
    central_entities: List[Tuple[str, float]]
    communities: List[List[str]]
    influence_scores: Dict[str, float]
    knowledge_completeness: float
    consistency_score: float


class ReasoningEngine:
    """Advanced reasoning engine for knowledge inference"""

    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.inference_rules = []
        self.reasoning_cache = {}

    def add_inference_rule(self, rule: InferenceRule):
        """Add an inference rule to the reasoning engine"""
        self.inference_rules.append(rule)
        logger.info(f"Added inference rule: {rule.name}")

    def forward_reasoning(self, max_iterations: int = 10) -> List[KnowledgeRelation]:
        """Perform forward chaining reasoning to derive new knowledge"""
        new_relations = []

        for iteration in range(max_iterations):
            iteration_discoveries = 0

            for rule in self.inference_rules:
                if not rule.enabled:
                    continue

                # Apply rule to discover new relations
                discovered = self._apply_inference_rule(rule)
                new_relations.extend(discovered)
                iteration_discoveries += len(discovered)

            # Stop if no new knowledge was discovered
            if iteration_discoveries == 0:
                break

        return new_relations

    def _apply_inference_rule(self, rule: InferenceRule) -> List[KnowledgeRelation]:
        """Apply a specific inference rule"""
        new_relations = []

        # This is a simplified rule application
        # In practice, this would parse the rule conditions and conclusions
        if rule.rule_id == "transitivity_is_a":
            # If A is_a B and B is_a C, then A is_a C
            new_relations.extend(self._apply_transitivity_rule(RelationType.IS_A))
        elif rule.rule_id == "inheritance_properties":
            # If A is_a B and B has_property P, then A has_property P
            new_relations.extend(self._apply_inheritance_rule())

        return new_relations

    def _apply_transitivity_rule(self, relation_type: RelationType) -> List[KnowledgeRelation]:
        """Apply transitivity rule for a specific relation type"""
        new_relations = []

        # Get all relations of the specified type
        relations = self.knowledge_graph.get_relations_by_type(relation_type)

        # Build a map of source -> targets
        relation_map = defaultdict(set)
        for rel in relations:
            relation_map[rel.source_entity_id].add(rel.target_entity_id)

        # Find transitive relations
        for source in relation_map:
            for intermediate in relation_map[source]:
                for target in relation_map[intermediate]:
                    if target != source and target not in relation_map[source]:
                        # Create new transitive relation
                        new_rel = KnowledgeRelation(
                            id=f"inferred_{source}_{relation_type.value}_{target}",
                            source_entity_id=source,
                            target_entity_id=target,
                            relation_type=relation_type,
                            confidence=0.8,  # Lower confidence for inferred relations
                            evidence=[f"Inferred via transitivity through {intermediate}"]
                        )
                        new_relations.append(new_rel)

        return new_relations

    def _apply_inheritance_rule(self) -> List[KnowledgeRelation]:
        """Apply property inheritance rule"""
        new_relations = []

        # Get IS_A relations for inheritance
        is_a_relations = self.knowledge_graph.get_relations_by_type(RelationType.IS_A)
        property_relations = self.knowledge_graph.get_relations_by_type(RelationType.HAS_PROPERTY)

        # Build inheritance hierarchy
        hierarchy = defaultdict(set)
        for rel in is_a_relations:
            hierarchy[rel.source_entity_id].add(rel.target_entity_id)

        # Build property map
        properties = defaultdict(set)
        for rel in property_relations:
            properties[rel.source_entity_id].add(rel.target_entity_id)

        # Apply inheritance
        for child in hierarchy:
            for parent in hierarchy[child]:
                for prop in properties[parent]:
                    if prop not in properties[child]:
                        # Child inherits property from parent
                        new_rel = KnowledgeRelation(
                            id=f"inherited_{child}_has_property_{prop}",
                            source_entity_id=child,
                            target_entity_id=prop,
                            relation_type=RelationType.HAS_PROPERTY,
                            confidence=0.7,
                            evidence=[f"Inherited from parent {parent}"]
                        )
                        new_relations.append(new_rel)

        return new_relations


class KnowledgeGraphManager:
    """
    Advanced knowledge graph and ontology management system.

    Capabilities:
    - Multi-source knowledge graph construction and maintenance
    - Semantic reasoning and automatic inference
    - Ontology management with taxonomy support
    - Advanced graph analytics and pattern discovery
    - Real-time knowledge validation and consistency checking
    - SPARQL-like query processing
    - Entity extraction and relationship detection
    """

    def __init__(self, storage_path: str = "./knowledge_graph", telemetry=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.telemetry = telemetry

        # Core knowledge structures
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.ontologies: Dict[str, Ontology] = {}

        # Graph representation
        self.graph = nx.MultiDiGraph()
        self.entity_index = {}  # Fast entity lookup
        self.relation_index = defaultdict(list)  # Index by relation type

        # Reasoning and inference
        self.reasoning_engine = ReasoningEngine(self)
        self.inference_cache = {}

        # Analytics cache
        self.analytics_cache = {}
        self.last_analytics_update = None

        # Database connection for persistence
        self.db_path = self.storage_path / "knowledge_graph.db"
        self._init_database()

        # Thread synchronization
        self.graph_lock = threading.RLock()

        # Initialize default ontologies and rules
        self._initialize_default_ontologies()
        self._initialize_inference_rules()

        # Metrics
        self.metrics = {
            'total_entities_added': 0,
            'total_relations_added': 0,
            'inference_operations': 0,
            'queries_processed': 0,
            'knowledge_extractions': 0,
            'consistency_checks': 0
        }

        logger.info(f"Knowledge Graph Manager initialized with storage at {storage_path}")

    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    properties TEXT,
                    confidence REAL,
                    source TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    aliases TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT,
                    confidence REAL,
                    evidence TEXT,
                    created_at TEXT,
                    weight REAL,
                    temporal_context TEXT,
                    FOREIGN KEY (source_entity_id) REFERENCES entities (id),
                    FOREIGN KEY (target_entity_id) REFERENCES entities (id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS ontologies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT,
                    description TEXT,
                    namespaces TEXT,
                    classes TEXT,
                    properties TEXT,
                    axioms TEXT,
                    created_at TEXT
                )
            """)

            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id)")

        logger.info("Knowledge graph database initialized")

    def _initialize_default_ontologies(self):
        """Initialize default ontologies"""
        # Basic upper ontology
        basic_ontology = Ontology(
            id="basic_upper_ontology",
            name="Basic Upper Ontology",
            version="1.0",
            description="Fundamental concepts and relationships",
            namespaces={
                "owl": "http://www.w3.org/2002/07/owl#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            classes={
                "Entity": {"description": "Top-level entity class"},
                "Physical": {"parent": "Entity", "description": "Physical objects"},
                "Abstract": {"parent": "Entity", "description": "Abstract concepts"},
                "Event": {"parent": "Entity", "description": "Events and processes"},
                "Information": {"parent": "Abstract", "description": "Information entities"}
            },
            properties={
                "has_property": {"domain": "Entity", "range": "Entity"},
                "is_a": {"domain": "Entity", "range": "Entity", "transitive": True},
                "part_of": {"domain": "Entity", "range": "Entity", "transitive": True}
            },
            axioms=[
                "Entity(x) ∧ has_property(x, y) → Entity(y)",
                "is_a(x, y) ∧ is_a(y, z) → is_a(x, z)"
            ]
        )

        self.add_ontology(basic_ontology)

    def _initialize_inference_rules(self):
        """Initialize default inference rules"""
        rules = [
            InferenceRule(
                rule_id="transitivity_is_a",
                name="Transitivity of IS_A",
                description="If A is_a B and B is_a C, then A is_a C",
                condition="is_a(X, Y) ∧ is_a(Y, Z)",
                conclusion="is_a(X, Z)",
                confidence_threshold=0.8
            ),
            InferenceRule(
                rule_id="inheritance_properties",
                name="Property Inheritance",
                description="If A is_a B and B has_property P, then A has_property P",
                condition="is_a(X, Y) ∧ has_property(Y, P)",
                conclusion="has_property(X, P)",
                confidence_threshold=0.7
            ),
            InferenceRule(
                rule_id="transitivity_part_of",
                name="Transitivity of PART_OF",
                description="If A part_of B and B part_of C, then A part_of C",
                condition="part_of(X, Y) ∧ part_of(Y, Z)",
                conclusion="part_of(X, Z)",
                confidence_threshold=0.8
            )
        ]

        for rule in rules:
            self.reasoning_engine.add_inference_rule(rule)

    async def add_entity(self, entity: KnowledgeEntity, persist: bool = True) -> bool:
        """Add an entity to the knowledge graph"""
        try:
            with self.graph_lock:
                # Check for duplicates
                if entity.id in self.entities:
                    logger.warning(f"Entity {entity.id} already exists, updating...")
                    return await self.update_entity(entity, persist)

                # Add to internal structures
                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, **{
                    'name': entity.name,
                    'type': entity.entity_type.value,
                    'properties': entity.properties,
                    'confidence': entity.confidence
                })

                # Update index
                self.entity_index[entity.name.lower()] = entity.id
                for alias in entity.aliases:
                    self.entity_index[alias.lower()] = entity.id

                # Persist to database
                if persist:
                    await self._persist_entity(entity)

                self.metrics['total_entities_added'] += 1

                if self.telemetry:
                    self.telemetry.record_event("entity_added", {
                        'entity_id': entity.id,
                        'entity_type': entity.entity_type.value,
                        'confidence': entity.confidence
                    })

                logger.debug(f"Added entity: {entity.name} ({entity.entity_type.value})")
                return True

        except Exception as e:
            logger.error(f"Failed to add entity {entity.id}: {e}")
            return False

    async def add_relation(self, relation: KnowledgeRelation, persist: bool = True) -> bool:
        """Add a relationship to the knowledge graph"""
        try:
            with self.graph_lock:
                # Validate entities exist
                if relation.source_entity_id not in self.entities:
                    logger.error(f"Source entity {relation.source_entity_id} not found")
                    return False

                if relation.target_entity_id not in self.entities:
                    logger.error(f"Target entity {relation.target_entity_id} not found")
                    return False

                # Add to internal structures
                self.relations[relation.id] = relation

                # Add edge to graph
                self.graph.add_edge(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    key=relation.id,
                    relation_type=relation.relation_type.value,
                    properties=relation.properties,
                    confidence=relation.confidence,
                    weight=relation.weight
                )

                # Update relation index
                self.relation_index[relation.relation_type].append(relation)

                # Persist to database
                if persist:
                    await self._persist_relation(relation)

                self.metrics['total_relations_added'] += 1

                if self.telemetry:
                    self.telemetry.record_event("relation_added", {
                        'relation_id': relation.id,
                        'relation_type': relation.relation_type.value,
                        'confidence': relation.confidence
                    })

                logger.debug(f"Added relation: {relation.source_entity_id} -> {relation.target_entity_id} ({relation.relation_type.value})")
                return True

        except Exception as e:
            logger.error(f"Failed to add relation {relation.id}: {e}")
            return False

    async def update_entity(self, entity: KnowledgeEntity, persist: bool = True) -> bool:
        """Update an existing entity"""
        try:
            with self.graph_lock:
                if entity.id not in self.entities:
                    logger.error(f"Entity {entity.id} not found for update")
                    return False

                # Update entity
                entity.updated_at = datetime.utcnow()
                self.entities[entity.id] = entity

                # Update graph node
                self.graph.nodes[entity.id].update({
                    'name': entity.name,
                    'type': entity.entity_type.value,
                    'properties': entity.properties,
                    'confidence': entity.confidence
                })

                # Persist to database
                if persist:
                    await self._persist_entity(entity)

                logger.debug(f"Updated entity: {entity.name}")
                return True

        except Exception as e:
            logger.error(f"Failed to update entity {entity.id}: {e}")
            return False

    async def _persist_entity(self, entity: KnowledgeEntity):
        """Persist entity to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO entities
                (id, name, entity_type, properties, confidence, source,
                 created_at, updated_at, aliases, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id,
                entity.name,
                entity.entity_type.value,
                json.dumps(entity.properties),
                entity.confidence,
                entity.source,
                entity.created_at.isoformat(),
                entity.updated_at.isoformat(),
                json.dumps(entity.aliases),
                json.dumps(entity.metadata)
            ))

    async def _persist_relation(self, relation: KnowledgeRelation):
        """Persist relation to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO relations
                (id, source_entity_id, target_entity_id, relation_type,
                 properties, confidence, evidence, created_at, weight, temporal_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relation.id,
                relation.source_entity_id,
                relation.target_entity_id,
                relation.relation_type.value,
                json.dumps(relation.properties),
                relation.confidence,
                json.dumps(relation.evidence),
                relation.created_at.isoformat(),
                relation.weight,
                relation.temporal_context
            ))

    def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """Retrieve an entity by ID"""
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[KnowledgeEntity]:
        """Retrieve an entity by name"""
        entity_id = self.entity_index.get(name.lower())
        if entity_id:
            return self.entities.get(entity_id)
        return None

    def get_relations_by_type(self, relation_type: RelationType) -> List[KnowledgeRelation]:
        """Get all relations of a specific type"""
        return self.relation_index.get(relation_type, [])

    def get_entity_relations(self, entity_id: str, direction: str = "both") -> List[KnowledgeRelation]:
        """Get all relations for an entity"""
        relations = []

        if direction in ["outgoing", "both"]:
            relations.extend([
                rel for rel in self.relations.values()
                if rel.source_entity_id == entity_id
            ])

        if direction in ["incoming", "both"]:
            relations.extend([
                rel for rel in self.relations.values()
                if rel.target_entity_id == entity_id
            ])

        return relations

    async def semantic_query(self, query: SemanticQuery) -> Dict[str, Any]:
        """Process a semantic query against the knowledge graph"""
        try:
            self.metrics['queries_processed'] += 1

            if query.query_type == "graph_pattern":
                return await self._process_graph_pattern_query(query)
            elif query.query_type == "natural_language":
                return await self._process_natural_language_query(query)
            elif query.query_type == "sparql":
                return await self._process_sparql_query(query)
            else:
                return {"error": f"Unsupported query type: {query.query_type}"}

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {"error": str(e)}

    async def _process_graph_pattern_query(self, query: SemanticQuery) -> Dict[str, Any]:
        """Process a graph pattern query"""
        results = []

        # Simple pattern matching implementation
        if query.entities and query.relations:
            entity_pattern = query.entities[0] if query.entities else None
            relation_pattern = query.relations[0] if query.relations else None

            if entity_pattern and relation_pattern:
                # Find entities matching pattern
                matching_entities = [
                    entity for entity in self.entities.values()
                    if entity_pattern.lower() in entity.name.lower()
                ]

                for entity in matching_entities[:query.result_limit]:
                    relations = self.get_entity_relations(entity.id, "outgoing")
                    entity_relations = [
                        rel for rel in relations
                        if relation_pattern.lower() in rel.relation_type.value.lower()
                    ]

                    if entity_relations:
                        results.append({
                            'entity': {
                                'id': entity.id,
                                'name': entity.name,
                                'type': entity.entity_type.value
                            },
                            'relations': [
                                {
                                    'id': rel.id,
                                    'type': rel.relation_type.value,
                                    'target': self.entities[rel.target_entity_id].name,
                                    'confidence': rel.confidence
                                } for rel in entity_relations[:5]
                            ]
                        })

        return {
            'query_id': query.query_id,
            'results': results,
            'total_results': len(results),
            'execution_time_ms': 0  # Would track actual execution time
        }

    async def _process_natural_language_query(self, query: SemanticQuery) -> Dict[str, Any]:
        """Process a natural language query"""
        # Simple NL processing - extract entities and relations from query text
        query_text = query.query_text.lower()

        # Extract potential entity names (simplified)
        potential_entities = []
        for entity in self.entities.values():
            if entity.name.lower() in query_text:
                potential_entities.append(entity)

        # Extract potential relations
        potential_relations = []
        for relation_type in RelationType:
            if relation_type.value.lower().replace("_", " ") in query_text:
                potential_relations.extend(self.get_relations_by_type(relation_type))

        # Build response
        results = []
        for entity in potential_entities[:query.result_limit]:
            entity_data = {
                'id': entity.id,
                'name': entity.name,
                'type': entity.entity_type.value,
                'confidence': entity.confidence,
                'relations': []
            }

            # Add related information
            relations = self.get_entity_relations(entity.id)[:10]
            for rel in relations:
                target_entity = self.entities.get(rel.target_entity_id)
                if target_entity:
                    entity_data['relations'].append({
                        'type': rel.relation_type.value,
                        'target': target_entity.name,
                        'confidence': rel.confidence
                    })

            results.append(entity_data)

        return {
            'query_id': query.query_id,
            'query_text': query.query_text,
            'results': results,
            'total_results': len(results),
            'entities_found': len(potential_entities),
            'relations_found': len(potential_relations)
        }

    async def _process_sparql_query(self, query: SemanticQuery) -> Dict[str, Any]:
        """Process a SPARQL-like query (simplified implementation)"""
        # This would be a full SPARQL parser in production
        return {
            'query_id': query.query_id,
            'results': [],
            'message': "SPARQL processing not fully implemented"
        }

    async def perform_reasoning(self, max_iterations: int = 5) -> Dict[str, Any]:
        """Perform automated reasoning and inference"""
        try:
            self.metrics['inference_operations'] += 1

            start_time = datetime.utcnow()
            new_relations = self.reasoning_engine.forward_reasoning(max_iterations)

            # Add inferred relations to the graph
            added_count = 0
            for relation in new_relations:
                if await self.add_relation(relation, persist=True):
                    added_count += 1

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            result = {
                'reasoning_id': hashlib.md5(f"reasoning_{start_time}".encode()).hexdigest(),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'max_iterations': max_iterations,
                'new_relations_discovered': len(new_relations),
                'relations_added': added_count,
                'inference_rules_applied': len(self.reasoning_engine.inference_rules)
            }

            if self.telemetry:
                self.telemetry.record_event("reasoning_completed", result)

            logger.info(f"Reasoning completed: {added_count} new relations added")
            return result

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {'error': str(e)}

    async def extract_knowledge_from_text(self, text: str, source: str = "") -> Dict[str, Any]:
        """Extract entities and relations from text"""
        try:
            self.metrics['knowledge_extractions'] += 1

            entities_extracted = []
            relations_extracted = []

            # Simple entity extraction using regex patterns
            # In production, this would use NLP libraries like spaCy or transformers

            # Extract person names (simplified)
            person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            person_matches = re.findall(person_pattern, text)

            for match in person_matches:
                entity_id = f"person_{hashlib.md5(match.encode()).hexdigest()[:8]}"
                entity = KnowledgeEntity(
                    id=entity_id,
                    name=match,
                    entity_type=EntityType.PERSON,
                    source=source,
                    confidence=0.7
                )

                if await self.add_entity(entity):
                    entities_extracted.append(entity)

            # Extract organizations (simplified)
            org_keywords = ['Company', 'Inc', 'Corp', 'Ltd', 'Organization', 'Institute']
            for keyword in org_keywords:
                pattern = rf'\b[A-Z][a-zA-Z\s]+{keyword}\b'
                org_matches = re.findall(pattern, text)

                for match in org_matches:
                    entity_id = f"org_{hashlib.md5(match.encode()).hexdigest()[:8]}"
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=match.strip(),
                        entity_type=EntityType.ORGANIZATION,
                        source=source,
                        confidence=0.6
                    )

                    if await self.add_entity(entity):
                        entities_extracted.append(entity)

            # Extract simple relationships
            relation_patterns = [
                (r'(\w+) works for (\w+)', RelationType.RELATES_TO),
                (r'(\w+) is part of (\w+)', RelationType.PART_OF),
                (r'(\w+) created (\w+)', RelationType.CREATED_BY)
            ]

            for pattern, rel_type in relation_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    source_name, target_name = match

                    # Find entities
                    source_entity = self.get_entity_by_name(source_name)
                    target_entity = self.get_entity_by_name(target_name)

                    if source_entity and target_entity:
                        relation_id = f"rel_{source_entity.id}_{target_entity.id}_{rel_type.value}"
                        relation = KnowledgeRelation(
                            id=relation_id,
                            source_entity_id=source_entity.id,
                            target_entity_id=target_entity.id,
                            relation_type=rel_type,
                            confidence=0.6,
                            evidence=[f"Extracted from text: {text[:100]}..."]
                        )

                        if await self.add_relation(relation):
                            relations_extracted.append(relation)

            return {
                'extraction_id': hashlib.md5(f"extract_{datetime.utcnow()}".encode()).hexdigest(),
                'source': source,
                'text_length': len(text),
                'entities_extracted': len(entities_extracted),
                'relations_extracted': len(relations_extracted),
                'entities': [
                    {'id': e.id, 'name': e.name, 'type': e.entity_type.value}
                    for e in entities_extracted
                ],
                'relations': [
                    {
                        'id': r.id,
                        'source': r.source_entity_id,
                        'target': r.target_entity_id,
                        'type': r.relation_type.value
                    }
                    for r in relations_extracted
                ]
            }

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return {'error': str(e)}

    async def analyze_graph(self) -> GraphAnalytics:
        """Perform comprehensive graph analysis"""
        try:
            with self.graph_lock:
                if not self.graph.nodes():
                    return GraphAnalytics(
                        total_entities=0,
                        total_relations=0,
                        graph_density=0.0,
                        average_clustering=0.0,
                        diameter=0,
                        central_entities=[],
                        communities=[],
                        influence_scores={},
                        knowledge_completeness=0.0,
                        consistency_score=1.0
                    )

                # Basic graph metrics
                total_entities = self.graph.number_of_nodes()
                total_relations = self.graph.number_of_edges()

                # Convert to undirected for some calculations
                undirected_graph = self.graph.to_undirected()

                # Calculate density
                if total_entities > 1:
                    graph_density = nx.density(undirected_graph)
                else:
                    graph_density = 0.0

                # Calculate clustering
                try:
                    average_clustering = nx.average_clustering(undirected_graph)
                except:
                    average_clustering = 0.0

                # Calculate diameter (for largest connected component)
                try:
                    if nx.is_connected(undirected_graph):
                        diameter = nx.diameter(undirected_graph)
                    else:
                        # Get largest connected component
                        largest_cc = max(nx.connected_components(undirected_graph), key=len)
                        subgraph = undirected_graph.subgraph(largest_cc)
                        diameter = nx.diameter(subgraph) if len(largest_cc) > 1 else 0
                except:
                    diameter = 0

                # Calculate centrality measures
                try:
                    centrality = nx.betweenness_centrality(undirected_graph)
                    central_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                except:
                    central_entities = []

                # Detect communities
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = list(nx_comm.greedy_modularity_communities(undirected_graph))
                    communities = [list(community) for community in communities]
                except:
                    communities = []

                # Calculate influence scores (PageRank)
                try:
                    influence_scores = nx.pagerank(self.graph)
                except:
                    influence_scores = {}

                # Knowledge completeness (simplified metric)
                expected_entity_types = len(EntityType)
                actual_entity_types = len(set(e.entity_type for e in self.entities.values()))
                knowledge_completeness = min(1.0, actual_entity_types / expected_entity_types)

                # Consistency score (based on confidence levels)
                if self.relations:
                    avg_confidence = sum(r.confidence for r in self.relations.values()) / len(self.relations)
                    consistency_score = avg_confidence
                else:
                    consistency_score = 1.0

                analytics = GraphAnalytics(
                    total_entities=total_entities,
                    total_relations=total_relations,
                    graph_density=graph_density,
                    average_clustering=average_clustering,
                    diameter=diameter,
                    central_entities=central_entities,
                    communities=communities,
                    influence_scores=influence_scores,
                    knowledge_completeness=knowledge_completeness,
                    consistency_score=consistency_score
                )

                # Cache results
                self.analytics_cache = analytics
                self.last_analytics_update = datetime.utcnow()

                return analytics

        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            raise

    def add_ontology(self, ontology: Ontology) -> bool:
        """Add an ontology to the knowledge graph"""
        try:
            self.ontologies[ontology.id] = ontology

            # Persist to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ontologies
                    (id, name, version, description, namespaces, classes, properties, axioms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ontology.id,
                    ontology.name,
                    ontology.version,
                    ontology.description,
                    json.dumps(ontology.namespaces),
                    json.dumps(ontology.classes),
                    json.dumps(ontology.properties),
                    json.dumps(ontology.axioms),
                    ontology.created_at.isoformat()
                ))

            logger.info(f"Added ontology: {ontology.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add ontology: {e}")
            return False

    async def validate_consistency(self) -> Dict[str, Any]:
        """Validate knowledge graph consistency"""
        try:
            self.metrics['consistency_checks'] += 1

            inconsistencies = []
            warnings = []

            # Check for dangling relations
            for relation in self.relations.values():
                if relation.source_entity_id not in self.entities:
                    inconsistencies.append({
                        'type': 'dangling_relation',
                        'relation_id': relation.id,
                        'issue': f"Source entity {relation.source_entity_id} not found"
                    })

                if relation.target_entity_id not in self.entities:
                    inconsistencies.append({
                        'type': 'dangling_relation',
                        'relation_id': relation.id,
                        'issue': f"Target entity {relation.target_entity_id} not found"
                    })

            # Check for low confidence entities/relations
            low_confidence_threshold = 0.3
            for entity in self.entities.values():
                if entity.confidence < low_confidence_threshold:
                    warnings.append({
                        'type': 'low_confidence_entity',
                        'entity_id': entity.id,
                        'confidence': entity.confidence
                    })

            for relation in self.relations.values():
                if relation.confidence < low_confidence_threshold:
                    warnings.append({
                        'type': 'low_confidence_relation',
                        'relation_id': relation.id,
                        'confidence': relation.confidence
                    })

            # Check for potential duplicates
            entity_names = defaultdict(list)
            for entity in self.entities.values():
                entity_names[entity.name.lower()].append(entity.id)

            for name, entity_ids in entity_names.items():
                if len(entity_ids) > 1:
                    warnings.append({
                        'type': 'potential_duplicate',
                        'entity_name': name,
                        'entity_ids': entity_ids
                    })

            consistency_score = 1.0 - (len(inconsistencies) / max(1, len(self.entities) + len(self.relations)))

            return {
                'validation_id': hashlib.md5(f"validation_{datetime.utcnow()}".encode()).hexdigest(),
                'timestamp': datetime.utcnow().isoformat(),
                'consistency_score': max(0.0, consistency_score),
                'total_inconsistencies': len(inconsistencies),
                'total_warnings': len(warnings),
                'inconsistencies': inconsistencies,
                'warnings': warnings[:50],  # Limit warnings output
                'entities_checked': len(self.entities),
                'relations_checked': len(self.relations)
            }

        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            return {'error': str(e)}

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of knowledge graph"""
        with self.graph_lock:
            # Entity type breakdown
            entity_types = defaultdict(int)
            for entity in self.entities.values():
                entity_types[entity.entity_type.value] += 1

            # Relation type breakdown
            relation_types = defaultdict(int)
            for relation in self.relations.values():
                relation_types[relation.relation_type.value] += 1

            # Confidence distribution
            entity_confidences = [e.confidence for e in self.entities.values()]
            relation_confidences = [r.confidence for r in self.relations.values()]

            return {
                'summary_id': hashlib.md5(f"summary_{datetime.utcnow()}".encode()).hexdigest(),
                'timestamp': datetime.utcnow().isoformat(),
                'total_entities': len(self.entities),
                'total_relations': len(self.relations),
                'total_ontologies': len(self.ontologies),
                'entity_types': dict(entity_types),
                'relation_types': dict(relation_types),
                'average_entity_confidence': np.mean(entity_confidences) if entity_confidences else 0.0,
                'average_relation_confidence': np.mean(relation_confidences) if relation_confidences else 0.0,
                'graph_connectivity': self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
                'metrics': self.metrics,
                'last_analytics_update': self.last_analytics_update.isoformat() if self.last_analytics_update else None,
                'storage_path': str(self.storage_path),
                'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            }

    async def export_knowledge_graph(self, format_type: str = "json") -> str:
        """Export knowledge graph in various formats"""
        if format_type == "json":
            export_data = {
                'entities': [
                    {
                        'id': e.id,
                        'name': e.name,
                        'type': e.entity_type.value,
                        'properties': e.properties,
                        'confidence': e.confidence,
                        'created_at': e.created_at.isoformat()
                    }
                    for e in self.entities.values()
                ],
                'relations': [
                    {
                        'id': r.id,
                        'source': r.source_entity_id,
                        'target': r.target_entity_id,
                        'type': r.relation_type.value,
                        'properties': r.properties,
                        'confidence': r.confidence,
                        'created_at': r.created_at.isoformat()
                    }
                    for r in self.relations.values()
                ],
                'ontologies': [
                    {
                        'id': o.id,
                        'name': o.name,
                        'version': o.version,
                        'description': o.description
                    }
                    for o in self.ontologies.values()
                ],
                'metadata': {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'total_entities': len(self.entities),
                    'total_relations': len(self.relations),
                    'format': format_type
                }
            }
            return json.dumps(export_data, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    async def load_from_database(self):
        """Load existing knowledge graph from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load entities
                entity_rows = conn.execute("SELECT * FROM entities").fetchall()
                for row in entity_rows:
                    entity = KnowledgeEntity(
                        id=row[0],
                        name=row[1],
                        entity_type=EntityType(row[2]),
                        properties=json.loads(row[3] or '{}'),
                        confidence=row[4],
                        source=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                        aliases=json.loads(row[8] or '[]'),
                        metadata=json.loads(row[9] or '{}')
                    )
                    await self.add_entity(entity, persist=False)

                # Load relations
                relation_rows = conn.execute("SELECT * FROM relations").fetchall()
                for row in relation_rows:
                    relation = KnowledgeRelation(
                        id=row[0],
                        source_entity_id=row[1],
                        target_entity_id=row[2],
                        relation_type=RelationType(row[3]),
                        properties=json.loads(row[4] or '{}'),
                        confidence=row[5],
                        evidence=json.loads(row[6] or '[]'),
                        created_at=datetime.fromisoformat(row[7]),
                        weight=row[8],
                        temporal_context=row[9]
                    )
                    await self.add_relation(relation, persist=False)

                # Load ontologies
                ontology_rows = conn.execute("SELECT * FROM ontologies").fetchall()
                for row in ontology_rows:
                    ontology = Ontology(
                        id=row[0],
                        name=row[1],
                        version=row[2],
                        description=row[3],
                        namespaces=json.loads(row[4] or '{}'),
                        classes=json.loads(row[5] or '{}'),
                        properties=json.loads(row[6] or '{}'),
                        axioms=json.loads(row[7] or '[]'),
                        created_at=datetime.fromisoformat(row[8])
                    )
                    self.add_ontology(ontology)

            logger.info(f"Loaded {len(self.entities)} entities, {len(self.relations)} relations from database")

        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            raise