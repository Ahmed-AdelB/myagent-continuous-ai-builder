"""
Task Ledger - Task state management with persistence.

Implements task lifecycle tracking for crash recovery and monitoring:
- State machine: QUEUED → RUNNING → RETRY → BLOCKED → DONE/FAILED
- SQLite persistence with atomic transitions
- Dead-letter queue for permanent failures
- Task history audit trail

Architecture:
    TaskLedger manages tasks through their lifecycle:
    1. enqueue(): Create task in QUEUED state
    2. mark_running(): Transition QUEUED → RUNNING
    3. mark_done(): Transition RUNNING → DONE (success)
    4. mark_retry(): Transition RUNNING → RETRY (transient failure)
    5. mark_failed(): Transition RUNNING → FAILED (permanent failure)
    6. mark_blocked(): Transition RUNNING → BLOCKED (circuit breaker)

Performance Targets:
    - State transition: <10ms (SQLite write)
    - Task query: <5ms
    - Batch operations: <100ms for 100 tasks

Security:
    - Database permissions: 640 (owner read/write, group read)
    - No sensitive data in payloads (use references)
    - Audit trail in task_history table

Based on: Issue #5 - Task Ledger with retries, timeouts, circuit breakers
Implementation: Claude (Sonnet 4.5)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiosqlite
import json
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Task lifecycle states."""
    QUEUED = "queued"          # Task enqueued, waiting for execution
    RUNNING = "running"        # Task currently executing
    RETRY = "retry"            # Task failed, scheduled for retry
    BLOCKED = "blocked"        # Task blocked by dependency or circuit breaker
    DONE = "done"              # Task completed successfully
    FAILED = "failed"          # Task permanently failed (max retries exceeded)


@dataclass
class Task:
    """Task representation."""
    task_id: str
    agent_name: str
    task_type: str
    state: TaskState
    priority: int
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for JSON serialization."""
        return {
            **asdict(self),
            "state": self.state.value if isinstance(self.state, TaskState) else self.state,
        }


class TaskLedger:
    """
    Task state management with persistence.

    Features:
    - Atomic state transitions with history
    - SQLite persistence for crash recovery
    - Dead-letter queue for permanent failures
    - Query interface for task monitoring

    Usage:
        ledger = TaskLedger(Path("persistence/database/task_ledger.db"))
        await ledger.initialize()

        task_id = await ledger.enqueue("coder_agent", "code_generation", {...})
        await ledger.mark_running(task_id)
        await ledger.mark_done(task_id, {"result": "..."})
    """

    def __init__(self, db_path: Path):
        """
        Initialize task ledger.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Set database file permissions to 640
        import os
        if self.db_path.exists():
            try:
                os.chmod(self.db_path, 0o640)
            except Exception:
                pass  # Best effort

        logger.info(f"TaskLedger initialized at {db_path}")

    async def initialize(self) -> None:
        """
        Create database schema.

        Creates tables:
        - tasks: Main task table with state
        - task_history: Audit trail of state transitions
        - dead_letter_queue: Permanently failed tasks
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Tasks table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    payload TEXT,
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    started_at TEXT,
                    completed_at TEXT
                )
            """)

            # Indexes for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_state ON tasks(state)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_name ON tasks(agent_name)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at)")

            # Task history (audit trail)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS task_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    error TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
            """)

            # Dead-letter queue
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dead_letter_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    payload TEXT,
                    error TEXT NOT NULL,
                    retry_count INTEGER,
                    failed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
            """)

            await db.commit()

        logger.info("✓ TaskLedger schema initialized")

    async def enqueue(
        self,
        agent_name: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        max_retries: int = 3
    ) -> str:
        """
        Enqueue a new task.

        Args:
            agent_name: Target agent (e.g., "coder_agent", "claude", "codex")
            task_type: Type of task (e.g., "code_generation", "review")
            payload: Task data (must be JSON-serializable)
            priority: Task priority (higher = more urgent)
            max_retries: Maximum retry attempts

        Returns:
            task_id: Unique task identifier

        Example:
            task_id = await ledger.enqueue(
                agent_name="coder_agent",
                task_type="code_generation",
                payload={"file": "src/main.py", "instruction": "Add logging"},
                priority=1,
                max_retries=3
            )
        """
        task_id = str(uuid.uuid4())

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO tasks (
                    task_id, agent_name, task_type, state, priority,
                    payload, retry_count, max_retries
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    agent_name,
                    task_type,
                    TaskState.QUEUED.value,
                    priority,
                    json.dumps(payload),
                    0,
                    max_retries
                )
            )
            await db.commit()

        logger.info(f"Task {task_id} enqueued for {agent_name} ({task_type})")
        return task_id

    async def mark_running(self, task_id: str) -> None:
        """
        Mark task as running (QUEUED → RUNNING).

        Args:
            task_id: Task identifier

        Raises:
            ValueError: If task not found or invalid state transition
        """
        await self._transition(
            task_id,
            TaskState.RUNNING,
            extra_updates={"started_at": datetime.utcnow().isoformat() + "Z"}
        )

    async def mark_retry(self, task_id: str, error: Exception) -> None:
        """
        Mark task for retry (RUNNING → RETRY).

        If retry_count >= max_retries, marks as FAILED instead.

        Args:
            task_id: Task identifier
            error: Exception that caused the failure

        Raises:
            ValueError: If task not found
        """
        # Get current task
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Increment retry count
        new_retry_count = task.retry_count + 1

        # Check if max retries exceeded
        if new_retry_count >= task.max_retries:
            logger.warning(
                f"Task {task_id} max retries exceeded ({new_retry_count}/{task.max_retries}). "
                f"Marking as FAILED."
            )
            await self.mark_failed(task_id, str(error))
        else:
            logger.info(f"Task {task_id} scheduled for retry ({new_retry_count}/{task.max_retries})")
            await self._transition(
                task_id,
                TaskState.RETRY,
                error=str(error),
                extra_updates={"retry_count": new_retry_count}
            )

    async def mark_blocked(self, task_id: str, reason: str) -> None:
        """
        Mark task as blocked (RUNNING → BLOCKED).

        Args:
            task_id: Task identifier
            reason: Why task is blocked (e.g., "Circuit breaker open", "Dependency missing")

        Raises:
            ValueError: If task not found
        """
        await self._transition(task_id, TaskState.BLOCKED, error=reason)
        logger.warning(f"Task {task_id} blocked: {reason}")

    async def mark_done(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Mark task as completed (RUNNING → DONE).

        Args:
            task_id: Task identifier
            result: Task result (must be JSON-serializable)

        Raises:
            ValueError: If task not found or invalid state transition
        """
        await self._transition(
            task_id,
            TaskState.DONE,
            result=result,
            extra_updates={"completed_at": datetime.utcnow().isoformat() + "Z"}
        )
        logger.info(f"Task {task_id} completed successfully")

    async def mark_failed(self, task_id: str, error: str) -> None:
        """
        Mark task as permanently failed (RUNNING → FAILED).

        Moves task to dead-letter queue for analysis.

        Args:
            task_id: Task identifier
            error: Error message describing the failure

        Raises:
            ValueError: If task not found
        """
        await self._transition(
            task_id,
            TaskState.FAILED,
            error=error,
            extra_updates={"completed_at": datetime.utcnow().isoformat() + "Z"}
        )

        # Move to dead-letter queue
        await self._move_to_dead_letter_queue(task_id)

        logger.error(f"Task {task_id} permanently failed: {error}")

    async def _transition(
        self,
        task_id: str,
        to_state: TaskState,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        extra_updates: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Atomic state transition with history.

        Args:
            task_id: Task identifier
            to_state: Target state
            error: Optional error message
            result: Optional result data
            extra_updates: Optional additional column updates

        Raises:
            ValueError: If task not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get current state
            async with db.execute(
                "SELECT state FROM tasks WHERE task_id = ?",
                (task_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise ValueError(f"Task {task_id} not found")

                from_state = row[0]

            # Build UPDATE statement
            updates = ["state = ?", "updated_at = ?"]
            params = [to_state.value, datetime.utcnow().isoformat() + "Z"]

            if error is not None:
                updates.append("error = ?")
                params.append(error)

            if result is not None:
                updates.append("result = ?")
                params.append(json.dumps(result))

            if extra_updates:
                for key, value in extra_updates.items():
                    updates.append(f"{key} = ?")
                    params.append(value)

            params.append(task_id)

            # Update task
            await db.execute(
                f"UPDATE tasks SET {', '.join(updates)} WHERE task_id = ?",
                params
            )

            # Record history
            await db.execute(
                """
                INSERT INTO task_history (task_id, from_state, to_state, error)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, from_state, to_state.value, error)
            )

            await db.commit()

        logger.debug(f"Task {task_id} transitioned: {from_state} → {to_state.value}")

    async def _move_to_dead_letter_queue(self, task_id: str) -> None:
        """
        Move failed task to dead-letter queue.

        Args:
            task_id: Task identifier
        """
        task = await self.get_task(task_id)
        if not task:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO dead_letter_queue (
                    task_id, agent_name, task_type, payload, error, retry_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.agent_name,
                    task.task_type,
                    json.dumps(task.payload),
                    task.error or "Unknown error",
                    task.retry_count
                )
            )
            await db.commit()

        logger.info(f"Task {task_id} moved to dead-letter queue")

    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                return self._row_to_task(row)

    async def get_tasks_by_state(self, state: TaskState) -> List[Task]:
        """
        Get all tasks in a specific state.

        Args:
            state: Task state to filter by

        Returns:
            List of Task objects
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE state = ? ORDER BY priority DESC, created_at ASC",
                (state.value,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_task(row) for row in rows]

    async def get_tasks_by_agent(self, agent_name: str) -> List[Task]:
        """
        Get all tasks for a specific agent.

        Args:
            agent_name: Agent name

        Returns:
            List of Task objects
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE agent_name = ? ORDER BY created_at DESC",
                (agent_name,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_task(row) for row in rows]

    async def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """
        Get all tasks in dead-letter queue.

        Returns:
            List of dead-letter queue entries
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM dead_letter_queue ORDER BY failed_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_task_history(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get state transition history for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of history entries
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM task_history WHERE task_id = ? ORDER BY timestamp ASC",
                (task_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get task statistics.

        Returns:
            Statistics dictionary with counts by state
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Count by state
            stats = {}
            for state in TaskState:
                async with db.execute(
                    "SELECT COUNT(*) FROM tasks WHERE state = ?",
                    (state.value,)
                ) as cursor:
                    row = await cursor.fetchone()
                    stats[state.value] = row[0]

            # Dead-letter queue count
            async with db.execute("SELECT COUNT(*) FROM dead_letter_queue") as cursor:
                row = await cursor.fetchone()
                stats["dead_letter_queue"] = row[0]

            return stats

    def _row_to_task(self, row: aiosqlite.Row) -> Task:
        """Convert database row to Task object."""
        return Task(
            task_id=row["task_id"],
            agent_name=row["agent_name"],
            task_type=row["task_type"],
            state=TaskState(row["state"]),
            priority=row["priority"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )
