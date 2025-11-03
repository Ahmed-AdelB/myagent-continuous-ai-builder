"""
Project Ledger - Complete history and version control for the continuous development
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import sqlite3
from loguru import logger


@dataclass
class CodeVersion:
    """Represents a version of code"""
    id: str
    file_path: str
    content: str
    hash: str
    timestamp: datetime
    iteration: int
    agent: str
    reason: str
    parent_version: Optional[str] = None
    test_results: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "hash": self.hash,
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
            "agent": self.agent,
            "reason": self.reason,
            "parent_version": self.parent_version,
            "test_results": self.test_results,
            "performance_metrics": self.performance_metrics
        }


class ProjectLedger:
    """
    Maintains complete history of all code versions, changes, and decisions
    Acts as the project's permanent memory
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.db_path = Path(f"persistence/database/{project_name}_ledger.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self.versions: Dict[str, CodeVersion] = {}
        self.current_versions: Dict[str, str] = {}  # file_path -> current version id
        self.decision_log: List[Dict] = []

        self._initialize_database()
        self._load_existing_data()

    def _initialize_database(self):
        """Initialize the SQLite database"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_versions (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                agent TEXT NOT NULL,
                reason TEXT,
                parent_version TEXT,
                test_results TEXT,
                performance_metrics TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_versions (
                file_path TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                agent TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                description TEXT NOT NULL,
                rationale TEXT,
                outcome TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                test_name TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                execution_time REAL,
                code_version TEXT
            )
        """)

        self.conn.commit()
        logger.info(f"Initialized project ledger database: {self.db_path}")

    def _load_existing_data(self):
        """Load existing data from database"""
        cursor = self.conn.cursor()

        # Load current versions
        cursor.execute("SELECT file_path, version_id FROM current_versions")
        for row in cursor.fetchall():
            self.current_versions[row[0]] = row[1]

        # Load recent decisions
        cursor.execute("""
            SELECT timestamp, iteration, agent, decision_type, description, rationale, outcome
            FROM decisions
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        for row in cursor.fetchall():
            self.decision_log.append({
                "timestamp": row[0],
                "iteration": row[1],
                "agent": row[2],
                "decision_type": row[3],
                "description": row[4],
                "rationale": row[5],
                "outcome": row[6]
            })

    def save_code_version(
        self,
        file_path: str,
        content: str,
        iteration: int,
        agent: str,
        reason: str,
        test_results: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None
    ) -> CodeVersion:
        """Save a new version of code"""
        # Generate hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check if this exact version already exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM code_versions WHERE hash = ?", (content_hash,))
        existing = cursor.fetchone()

        if existing:
            logger.info(f"Code version already exists: {existing[0]}")
            return self.get_version(existing[0])

        # Create new version
        version_id = f"v_{iteration}_{agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get parent version
        parent_version = self.current_versions.get(file_path)

        version = CodeVersion(
            id=version_id,
            file_path=file_path,
            content=content,
            hash=content_hash,
            timestamp=datetime.now(),
            iteration=iteration,
            agent=agent,
            reason=reason,
            parent_version=parent_version,
            test_results=test_results,
            performance_metrics=performance_metrics
        )

        # Save to database
        cursor.execute("""
            INSERT INTO code_versions
            (id, file_path, content, hash, timestamp, iteration, agent, reason,
             parent_version, test_results, performance_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version.id,
            version.file_path,
            version.content,
            version.hash,
            version.timestamp.isoformat(),
            version.iteration,
            version.agent,
            version.reason,
            version.parent_version,
            json.dumps(test_results) if test_results else None,
            json.dumps(performance_metrics) if performance_metrics else None
        ))

        # Update current version
        cursor.execute("""
            INSERT OR REPLACE INTO current_versions (file_path, version_id, updated_at)
            VALUES (?, ?, ?)
        """, (file_path, version_id, datetime.now().isoformat()))

        self.conn.commit()

        self.versions[version_id] = version
        self.current_versions[file_path] = version_id

        # Save to file system as well
        self._save_to_filesystem(version)

        logger.info(f"Saved new code version: {version_id} for {file_path}")
        return version

    def _save_to_filesystem(self, version: CodeVersion):
        """Save version to filesystem for easy access"""
        version_dir = Path(f"persistence/storage/code_versions/{version.iteration}/{version.agent}")
        version_dir.mkdir(parents=True, exist_ok=True)

        file_name = Path(version.file_path).name
        version_file = version_dir / f"{file_name}.{version.id}"

        with open(version_file, "w") as f:
            f.write(version.content)

        # Save metadata
        meta_file = version_dir / f"{file_name}.{version.id}.meta.json"
        with open(meta_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

    def get_version(self, version_id: str) -> Optional[CodeVersion]:
        """Get a specific code version"""
        if version_id in self.versions:
            return self.versions[version_id]

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT file_path, content, hash, timestamp, iteration, agent, reason,
                   parent_version, test_results, performance_metrics
            FROM code_versions
            WHERE id = ?
        """, (version_id,))

        row = cursor.fetchone()
        if row:
            version = CodeVersion(
                id=version_id,
                file_path=row[0],
                content=row[1],
                hash=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                iteration=row[4],
                agent=row[5],
                reason=row[6],
                parent_version=row[7],
                test_results=json.loads(row[8]) if row[8] else None,
                performance_metrics=json.loads(row[9]) if row[9] else None
            )
            self.versions[version_id] = version
            return version

        return None

    def get_current_version(self, file_path: str) -> Optional[CodeVersion]:
        """Get the current version of a file"""
        version_id = self.current_versions.get(file_path)
        if version_id:
            return self.get_version(version_id)
        return None

    def get_file_history(self, file_path: str) -> List[CodeVersion]:
        """Get complete history of a file"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, hash, timestamp, iteration, agent, reason,
                   parent_version, test_results, performance_metrics
            FROM code_versions
            WHERE file_path = ?
            ORDER BY timestamp DESC
        """, (file_path,))

        history = []
        for row in cursor.fetchall():
            version = CodeVersion(
                id=row[0],
                file_path=file_path,
                content=row[1],
                hash=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                iteration=row[4],
                agent=row[5],
                reason=row[6],
                parent_version=row[7],
                test_results=json.loads(row[8]) if row[8] else None,
                performance_metrics=json.loads(row[9]) if row[9] else None
            )
            history.append(version)

        return history

    def record_decision(
        self,
        iteration: int,
        agent: str,
        decision_type: str,
        description: str,
        rationale: str = None,
        outcome: str = None,
        metadata: Dict = None
    ):
        """Record a development decision"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO decisions
            (timestamp, iteration, agent, decision_type, description, rationale, outcome, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            iteration,
            agent,
            decision_type,
            description,
            rationale,
            outcome,
            json.dumps(metadata) if metadata else None
        ))
        self.conn.commit()

        self.decision_log.append({
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "agent": agent,
            "decision_type": decision_type,
            "description": description,
            "rationale": rationale,
            "outcome": outcome
        })

        logger.info(f"Recorded decision: {decision_type} - {description}")

    def record_test_result(
        self,
        iteration: int,
        test_name: str,
        status: str,
        execution_time: float,
        error_message: str = None,
        code_version: str = None
    ):
        """Record test execution result"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO test_history
            (timestamp, iteration, test_name, status, error_message, execution_time, code_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            iteration,
            test_name,
            status,
            error_message,
            execution_time,
            code_version
        ))
        self.conn.commit()

    def get_iteration_summary(self, iteration: int) -> Dict:
        """Get summary of all changes in an iteration"""
        cursor = self.conn.cursor()

        # Get code changes
        cursor.execute("""
            SELECT file_path, id, agent, reason
            FROM code_versions
            WHERE iteration = ?
        """, (iteration,))
        code_changes = [
            {"file": row[0], "version": row[1], "agent": row[2], "reason": row[3]}
            for row in cursor.fetchall()
        ]

        # Get decisions
        cursor.execute("""
            SELECT decision_type, description, agent
            FROM decisions
            WHERE iteration = ?
        """, (iteration,))
        decisions = [
            {"type": row[0], "description": row[1], "agent": row[2]}
            for row in cursor.fetchall()
        ]

        # Get test results
        cursor.execute("""
            SELECT test_name, status, execution_time
            FROM test_history
            WHERE iteration = ?
        """, (iteration,))
        test_results = [
            {"name": row[0], "status": row[1], "time": row[2]}
            for row in cursor.fetchall()
        ]

        return {
            "iteration": iteration,
            "code_changes": code_changes,
            "decisions": decisions,
            "test_results": test_results,
            "total_changes": len(code_changes)
        }

    def rollback_to_version(self, file_path: str, version_id: str):
        """Rollback a file to a specific version"""
        version = self.get_version(version_id)
        if not version or version.file_path != file_path:
            logger.error(f"Invalid rollback: version {version_id} for {file_path}")
            return False

        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE current_versions
            SET version_id = ?, updated_at = ?
            WHERE file_path = ?
        """, (version_id, datetime.now().isoformat(), file_path))
        self.conn.commit()

        self.current_versions[file_path] = version_id
        logger.info(f"Rolled back {file_path} to version {version_id}")
        return True

    def cleanup_old_versions(self, keep_last_n: int = 100):
        """Clean up old versions to save space"""
        cursor = self.conn.cursor()

        for file_path in self.current_versions.keys():
            # Get all versions for this file
            cursor.execute("""
                SELECT id FROM code_versions
                WHERE file_path = ?
                ORDER BY timestamp DESC
            """, (file_path,))

            versions = cursor.fetchall()

            # Keep current and last n versions
            current = self.current_versions[file_path]
            keep_versions = {current}
            keep_versions.update(v[0] for v in versions[:keep_last_n])

            # Delete old versions
            for version in versions[keep_last_n:]:
                if version[0] not in keep_versions:
                    cursor.execute("DELETE FROM code_versions WHERE id = ?", (version[0],))
                    logger.info(f"Cleaned up old version: {version[0]}")

        self.conn.commit()

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()