"""
Error Knowledge Graph - Learns from errors and builds a knowledge base of solutions
"""

import json
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import sqlite3
from collections import defaultdict
import hashlib
from loguru import logger


@dataclass
class ErrorNode:
    """Represents an error in the knowledge graph"""
    id: str
    error_type: str
    error_message: str
    error_hash: str
    file_path: Optional[str]
    line_number: Optional[int]
    first_occurrence: datetime
    last_occurrence: datetime
    occurrence_count: int
    context: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_hash": self.error_hash,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "first_occurrence": self.first_occurrence.isoformat(),
            "last_occurrence": self.last_occurrence.isoformat(),
            "occurrence_count": self.occurrence_count,
            "context": self.context
        }


@dataclass
class SolutionNode:
    """Represents a solution to an error"""
    id: str
    solution_type: str
    description: str
    code_changes: Dict
    success_rate: float
    avg_resolution_time: float
    usage_count: int
    created_by: str
    created_at: datetime

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "solution_type": self.solution_type,
            "description": self.description,
            "code_changes": self.code_changes,
            "success_rate": self.success_rate,
            "avg_resolution_time": self.avg_resolution_time,
            "usage_count": self.usage_count,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat()
        }


class ErrorKnowledgeGraph:
    """
    Maintains a knowledge graph of errors and their solutions.
    Learns patterns and improves over time.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.db_path = Path("persistence/database/error_knowledge.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.errors: Dict[str, ErrorNode] = {}
        self.solutions: Dict[str, SolutionNode] = {}
        self.error_patterns: Dict[str, List[str]] = defaultdict(list)
        self.solution_effectiveness: Dict[str, List[float]] = defaultdict(list)

        self._initialize_database()
        self._load_graph()

    def _initialize_database(self):
        """Initialize the SQLite database for persistent storage"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id TEXT PRIMARY KEY,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                error_hash TEXT NOT NULL,
                file_path TEXT,
                line_number INTEGER,
                first_occurrence TEXT NOT NULL,
                last_occurrence TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL,
                context TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solutions (
                id TEXT PRIMARY KEY,
                solution_type TEXT NOT NULL,
                description TEXT NOT NULL,
                code_changes TEXT NOT NULL,
                success_rate REAL NOT NULL,
                avg_resolution_time REAL NOT NULL,
                usage_count INTEGER NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_solution_links (
                error_id TEXT NOT NULL,
                solution_id TEXT NOT NULL,
                success_count INTEGER NOT NULL,
                failure_count INTEGER NOT NULL,
                avg_resolution_time REAL,
                last_used TEXT,
                PRIMARY KEY (error_id, solution_id),
                FOREIGN KEY (error_id) REFERENCES errors(id),
                FOREIGN KEY (solution_id) REFERENCES solutions(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_regex TEXT NOT NULL,
                error_ids TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        self.conn.commit()
        logger.info("Initialized error knowledge database")

    def _load_graph(self):
        """Load the knowledge graph from database"""
        cursor = self.conn.cursor()

        # Load errors
        cursor.execute("SELECT * FROM errors")
        for row in cursor.fetchall():
            error = ErrorNode(
                id=row[0],
                error_type=row[1],
                error_message=row[2],
                error_hash=row[3],
                file_path=row[4],
                line_number=row[5],
                first_occurrence=datetime.fromisoformat(row[6]),
                last_occurrence=datetime.fromisoformat(row[7]),
                occurrence_count=row[8],
                context=json.loads(row[9]) if row[9] else {}
            )
            self.errors[error.id] = error
            self.graph.add_node(error.id, type="error", data=error)

        # Load solutions
        cursor.execute("SELECT * FROM solutions")
        for row in cursor.fetchall():
            solution = SolutionNode(
                id=row[0],
                solution_type=row[1],
                description=row[2],
                code_changes=json.loads(row[3]),
                success_rate=row[4],
                avg_resolution_time=row[5],
                usage_count=row[6],
                created_by=row[7],
                created_at=datetime.fromisoformat(row[8])
            )
            self.solutions[solution.id] = solution
            self.graph.add_node(solution.id, type="solution", data=solution)

        # Load relationships
        cursor.execute("SELECT * FROM error_solution_links")
        for row in cursor.fetchall():
            self.graph.add_edge(
                row[0], row[1],
                success_count=row[2],
                failure_count=row[3],
                avg_resolution_time=row[4],
                last_used=row[5]
            )

        logger.info(f"Loaded knowledge graph with {len(self.errors)} errors and {len(self.solutions)} solutions")

    def add_error(
        self,
        error_type: str,
        error_message: str,
        file_path: str = None,
        line_number: int = None,
        context: Dict = None
    ) -> ErrorNode:
        """Add a new error to the knowledge graph"""
        # Generate error hash for deduplication
        error_hash = hashlib.md5(f"{error_type}:{error_message}".encode()).hexdigest()

        # Check if error already exists
        for error_id, error in self.errors.items():
            if error.error_hash == error_hash:
                # Update existing error
                error.last_occurrence = datetime.now()
                error.occurrence_count += 1
                self._update_error_in_db(error)
                logger.info(f"Updated existing error: {error_id}")
                return error

        # Create new error
        error_id = f"error_{len(self.errors) + 1}_{error_hash[:8]}"
        error = ErrorNode(
            id=error_id,
            error_type=error_type,
            error_message=error_message,
            error_hash=error_hash,
            file_path=file_path,
            line_number=line_number,
            first_occurrence=datetime.now(),
            last_occurrence=datetime.now(),
            occurrence_count=1,
            context=context or {}
        )

        # Save to database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO errors
            (id, error_type, error_message, error_hash, file_path, line_number,
             first_occurrence, last_occurrence, occurrence_count, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            error.id,
            error.error_type,
            error.error_message,
            error.error_hash,
            error.file_path,
            error.line_number,
            error.first_occurrence.isoformat(),
            error.last_occurrence.isoformat(),
            error.occurrence_count,
            json.dumps(error.context)
        ))
        self.conn.commit()

        # Add to graph
        self.errors[error_id] = error
        self.graph.add_node(error_id, type="error", data=error)

        # Check for patterns
        self._identify_error_patterns(error)

        logger.info(f"Added new error to knowledge graph: {error_id}")
        return error

    def add_solution(
        self,
        error_id: str,
        solution_type: str,
        description: str,
        code_changes: Dict,
        created_by: str,
        success: bool = True,
        resolution_time: float = 0.0
    ) -> SolutionNode:
        """Add a solution for an error"""
        # Check if similar solution exists
        solution_hash = hashlib.md5(
            f"{solution_type}:{json.dumps(code_changes, sort_keys=True)}".encode()
        ).hexdigest()

        for sol_id, solution in self.solutions.items():
            if hashlib.md5(
                f"{solution.solution_type}:{json.dumps(solution.code_changes, sort_keys=True)}".encode()
            ).hexdigest() == solution_hash:
                # Update existing solution
                self._update_solution_effectiveness(sol_id, error_id, success, resolution_time)
                return solution

        # Create new solution
        solution_id = f"solution_{len(self.solutions) + 1}_{solution_hash[:8]}"
        solution = SolutionNode(
            id=solution_id,
            solution_type=solution_type,
            description=description,
            code_changes=code_changes,
            success_rate=100.0 if success else 0.0,
            avg_resolution_time=resolution_time,
            usage_count=1,
            created_by=created_by,
            created_at=datetime.now()
        )

        # Save to database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO solutions
            (id, solution_type, description, code_changes, success_rate,
             avg_resolution_time, usage_count, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            solution.id,
            solution.solution_type,
            solution.description,
            json.dumps(solution.code_changes),
            solution.success_rate,
            solution.avg_resolution_time,
            solution.usage_count,
            solution.created_by,
            solution.created_at.isoformat()
        ))

        # Create link
        cursor.execute("""
            INSERT INTO error_solution_links
            (error_id, solution_id, success_count, failure_count,
             avg_resolution_time, last_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            error_id,
            solution_id,
            1 if success else 0,
            0 if success else 1,
            resolution_time,
            datetime.now().isoformat()
        ))
        self.conn.commit()

        # Add to graph
        self.solutions[solution_id] = solution
        self.graph.add_node(solution_id, type="solution", data=solution)
        self.graph.add_edge(
            error_id,
            solution_id,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            avg_resolution_time=resolution_time,
            last_used=datetime.now().isoformat()
        )

        logger.info(f"Added solution {solution_id} for error {error_id}")
        return solution

    def get_solutions_for_error(self, error_id: str) -> List[Tuple[SolutionNode, float]]:
        """Get ranked solutions for an error"""
        if error_id not in self.graph:
            return []

        solutions = []
        for neighbor in self.graph.neighbors(error_id):
            if self.graph.nodes[neighbor]["type"] == "solution":
                edge_data = self.graph[error_id][neighbor]
                solution = self.solutions[neighbor]

                # Calculate effectiveness score
                total = edge_data["success_count"] + edge_data["failure_count"]
                effectiveness = edge_data["success_count"] / total if total > 0 else 0.0

                solutions.append((solution, effectiveness))

        # Sort by effectiveness
        solutions.sort(key=lambda x: x[1], reverse=True)
        return solutions

    def find_similar_errors(self, error_type: str, error_message: str) -> List[ErrorNode]:
        """Find similar errors in the knowledge graph"""
        similar = []
        target_hash = hashlib.md5(f"{error_type}:{error_message}".encode()).hexdigest()

        for error_id, error in self.errors.items():
            # Check exact match
            if error.error_hash == target_hash:
                similar.append(error)
                continue

            # Check partial match
            similarity = self._calculate_similarity(error_message, error.error_message)
            if similarity > 0.7:  # 70% similarity threshold
                similar.append(error)

        return similar

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Simple Jaccard similarity
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _identify_error_patterns(self, error: ErrorNode):
        """Identify patterns in errors"""
        # Group errors by type
        self.error_patterns[error.error_type].append(error.id)

        # Check for recurring patterns
        if len(self.error_patterns[error.error_type]) >= 3:
            # Pattern detected
            logger.info(f"Pattern detected for error type: {error.error_type}")

    def _update_solution_effectiveness(
        self,
        solution_id: str,
        error_id: str,
        success: bool,
        resolution_time: float
    ):
        """Update solution effectiveness metrics"""
        solution = self.solutions[solution_id]
        edge_data = self.graph[error_id][solution_id]

        # Update counts
        if success:
            edge_data["success_count"] += 1
        else:
            edge_data["failure_count"] += 1

        # Update average resolution time
        total_count = edge_data["success_count"] + edge_data["failure_count"]
        edge_data["avg_resolution_time"] = (
            (edge_data["avg_resolution_time"] * (total_count - 1) + resolution_time) / total_count
        )
        edge_data["last_used"] = datetime.now().isoformat()

        # Update solution stats
        solution.usage_count += 1
        solution.success_rate = (edge_data["success_count"] / total_count) * 100
        solution.avg_resolution_time = edge_data["avg_resolution_time"]

        # Update database
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE error_solution_links
            SET success_count = ?, failure_count = ?, avg_resolution_time = ?, last_used = ?
            WHERE error_id = ? AND solution_id = ?
        """, (
            edge_data["success_count"],
            edge_data["failure_count"],
            edge_data["avg_resolution_time"],
            edge_data["last_used"],
            error_id,
            solution_id
        ))

        cursor.execute("""
            UPDATE solutions
            SET success_rate = ?, avg_resolution_time = ?, usage_count = ?
            WHERE id = ?
        """, (
            solution.success_rate,
            solution.avg_resolution_time,
            solution.usage_count,
            solution_id
        ))
        self.conn.commit()

    def _update_error_in_db(self, error: ErrorNode):
        """Update error in database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE errors
            SET last_occurrence = ?, occurrence_count = ?
            WHERE id = ?
        """, (
            error.last_occurrence.isoformat(),
            error.occurrence_count,
            error.id
        ))
        self.conn.commit()

    def get_learning_insights(self) -> Dict:
        """Get insights from the knowledge graph"""
        insights = {
            "total_errors": len(self.errors),
            "total_solutions": len(self.solutions),
            "most_common_errors": [],
            "most_effective_solutions": [],
            "error_patterns": {}
        }

        # Most common errors
        sorted_errors = sorted(
            self.errors.values(),
            key=lambda x: x.occurrence_count,
            reverse=True
        )[:5]
        insights["most_common_errors"] = [
            {"type": e.error_type, "count": e.occurrence_count}
            for e in sorted_errors
        ]

        # Most effective solutions
        sorted_solutions = sorted(
            self.solutions.values(),
            key=lambda x: x.success_rate,
            reverse=True
        )[:5]
        insights["most_effective_solutions"] = [
            {"type": s.solution_type, "success_rate": s.success_rate}
            for s in sorted_solutions
        ]

        # Error patterns
        for pattern_type, error_ids in self.error_patterns.items():
            if len(error_ids) >= 3:
                insights["error_patterns"][pattern_type] = len(error_ids)

        return insights

    def export_graph(self, path: str):
        """Export the knowledge graph for visualization"""
        import json
        graph_data = {
            "nodes": [],
            "edges": []
        }

        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            graph_data["nodes"].append({
                "id": node_id,
                "type": node_data["type"],
                "data": node_data["data"].to_dict()
            })

        for edge in self.graph.edges(data=True):
            graph_data["edges"].append({
                "source": edge[0],
                "target": edge[1],
                "data": edge[2]
            })

        with open(path, "w") as f:
            json.dump(graph_data, f, indent=2)

        logger.info(f"Exported knowledge graph to {path}")
        return graph_data