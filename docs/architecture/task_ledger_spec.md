# Task Ledger Architecture Specification

**Issue**: #5
**Priority**: CRITICAL (Sprint 1)
**Estimated Effort**: 3-5 days
**Status**: IN PROGRESS

## Overview

Implement orchestration hardening with task state management, exponential backoff retries, circuit breakers, and dead-letter queue for fault tolerance in 24/7 continuous operation.

## Problem Statement

Current orchestration lacks resilience:
- No task state tracking (can't recover from crashes)
- No retry logic (transient failures cause permanent task loss)
- No circuit breakers (cascade failures overwhelm system)
- No dead-letter queue (failed tasks disappear)

## Architecture

### Component Diagram

```
TriAgentSDLC Orchestrator
    ├── TaskLedger (core/resilience/task_ledger.py)
    │   ├── State Machine (QUEUED → RUNNING → RETRY → BLOCKED → DONE/FAILED)
    │   ├── SQLite persistence (persistence/database/task_ledger.db)
    │   ├── Atomic state transitions
    │   └── Dead-letter queue
    │
    ├── RetryStrategy (core/resilience/retry_strategy.py)
    │   ├── Exponential backoff with jitter
    │   ├── Configurable max retries (default: 3)
    │   └── Per-task retry counters
    │
    └── CircuitBreaker (core/resilience/circuit_breaker.py)
        ├── Per-agent circuit breakers
        ├── Trip threshold (5 consecutive failures)
        ├── States: CLOSED → OPEN → HALF_OPEN → CLOSED
        └── Auto-reset after cooldown (60s)
```

### Task State Machine

```python
class TaskState(Enum):
    """Task lifecycle states."""
    QUEUED = "queued"          # Task enqueued, waiting for execution
    RUNNING = "running"        # Task currently executing
    RETRY = "retry"            # Task failed, scheduled for retry
    BLOCKED = "blocked"        # Task blocked by dependency or circuit breaker
    DONE = "done"              # Task completed successfully
    FAILED = "failed"          # Task permanently failed (max retries exceeded)

# State transition rules:
# QUEUED → RUNNING (task starts)
# RUNNING → DONE (success)
# RUNNING → RETRY (transient failure, retries remaining)
# RUNNING → FAILED (permanent failure OR max retries exceeded)
# RUNNING → BLOCKED (circuit breaker open OR dependency missing)
# RETRY → RUNNING (retry attempt)
# BLOCKED → QUEUED (circuit breaker closed OR dependency resolved)
```

### Database Schema

```sql
-- persistence/database/task_ledger.db

CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    state TEXT NOT NULL,  -- TaskState enum value
    priority INTEGER DEFAULT 0,
    payload TEXT,  -- JSON serialized task data
    result TEXT,   -- JSON serialized result (if DONE)
    error TEXT,    -- Error message (if FAILED)
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_state ON tasks(state);
CREATE INDEX idx_agent_name ON tasks(agent_name);
CREATE INDEX idx_created_at ON tasks(created_at);

CREATE TABLE task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    error TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE dead_letter_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    payload TEXT,
    error TEXT NOT NULL,
    retry_count INTEGER,
    failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);
```

## Implementation Plan

### 1. TaskLedger (core/resilience/task_ledger.py)

```python
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiosqlite
import json
import uuid
from datetime import datetime

class TaskState(Enum):
    """Task lifecycle states."""
    QUEUED = "queued"
    RUNNING = "running"
    RETRY = "retry"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"

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
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class TaskLedger:
    """
    Task state management with persistence.

    Features:
    - Atomic state transitions with history
    - SQLite persistence for crash recovery
    - Dead-letter queue for permanent failures
    - Query interface for task monitoring
    """

    def __init__(self, db_path: Path):
        """Initialize task ledger."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Create database schema."""
        # Create tables (see schema above)
        pass

    async def enqueue(self, agent_name: str, task_type: str,
                     payload: Dict[str, Any], priority: int = 0,
                     max_retries: int = 3) -> str:
        """
        Enqueue a new task.

        Args:
            agent_name: Target agent
            task_type: Type of task (e.g., "code_generation", "review")
            payload: Task data
            priority: Task priority (higher = more urgent)
            max_retries: Maximum retry attempts

        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())
        # Insert into database with state=QUEUED
        return task_id

    async def mark_running(self, task_id: str) -> None:
        """Mark task as running (QUEUED → RUNNING)."""
        await self._transition(task_id, TaskState.RUNNING)

    async def mark_retry(self, task_id: str, error: Exception) -> None:
        """Mark task for retry (RUNNING → RETRY)."""
        # Increment retry_count
        # If retry_count >= max_retries: mark_failed() instead
        pass

    async def mark_blocked(self, task_id: str, reason: str) -> None:
        """Mark task as blocked (RUNNING → BLOCKED)."""
        await self._transition(task_id, TaskState.BLOCKED, error=reason)

    async def mark_done(self, task_id: str, result: Dict[str, Any]) -> None:
        """Mark task as completed (RUNNING → DONE)."""
        await self._transition(task_id, TaskState.DONE, result=result)

    async def mark_failed(self, task_id: str, error: str) -> None:
        """
        Mark task as permanently failed (RUNNING → FAILED).
        Moves task to dead-letter queue.
        """
        await self._transition(task_id, TaskState.FAILED, error=error)
        await self._move_to_dead_letter_queue(task_id)

    async def _transition(self, task_id: str, to_state: TaskState,
                         error: Optional[str] = None,
                         result: Optional[Dict[str, Any]] = None) -> None:
        """Atomic state transition with history."""
        # Update tasks table + insert into task_history
        pass

    async def _move_to_dead_letter_queue(self, task_id: str) -> None:
        """Move failed task to dead-letter queue."""
        pass

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        pass

    async def get_tasks_by_state(self, state: TaskState) -> List[Task]:
        """Get all tasks in a specific state."""
        pass

    async def get_dead_letter_queue(self) -> List[Task]:
        """Get all tasks in dead-letter queue."""
        pass
```

### 2. RetryStrategy (core/resilience/retry_strategy.py)

```python
import asyncio
import random
from typing import Callable, TypeVar, Optional
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class RetryConfig:
    """Retry strategy configuration."""
    max_retries: int = 3
    base_delay_ms: float = 100.0  # Initial delay in ms
    max_delay_ms: float = 30000.0  # Max delay 30s
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd

class RetryStrategy:
    """
    Exponential backoff with jitter.

    Delay formula:
        delay = min(base_delay * (exponential_base ** attempt), max_delay)
        if jitter: delay = delay * random(0.5, 1.5)

    Example (base_delay=100ms, exponential_base=2):
        Attempt 1: 100ms (+ jitter)
        Attempt 2: 200ms (+ jitter)
        Attempt 3: 400ms (+ jitter)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay_ms = min(
            self.config.base_delay_ms * (self.config.exponential_base ** attempt),
            self.config.max_delay_ms
        )

        if self.config.jitter:
            # Add ±50% jitter
            jitter_factor = random.uniform(0.5, 1.5)
            delay_ms *= jitter_factor

        return delay_ms / 1000.0  # Convert to seconds

    async def execute_with_retry(
        self,
        fn: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function with exponential backoff retry.

        Args:
            fn: Async function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                else:
                    return fn(*args, **kwargs)
            except Exception as exc:
                last_exception = exc

                if attempt < self.config.max_retries:
                    delay = self.calculate_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded
                    raise last_exception

        # Should never reach here
        raise last_exception
```

### 3. CircuitBreaker (core/resilience/circuit_breaker.py)

```python
import asyncio
from enum import Enum
from typing import Callable, TypeVar, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Consecutive failures before trip
    success_threshold: int = 2  # Consecutive successes to close from half-open
    timeout_seconds: float = 60.0  # Time before attempting half-open

class CircuitBreaker:
    """
    Circuit breaker pattern for agent fault tolerance.

    State transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After timeout_seconds
    - HALF_OPEN → CLOSED: After success_threshold consecutive successes
    - HALF_OPEN → OPEN: On any failure

    Usage:
        breaker = CircuitBreaker("coder_agent")
        result = await breaker.call(agent.process_task, task)
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None

    async def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.

        Args:
            fn: Async function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Original exception: If function fails
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpen(f"Circuit breaker '{self.name}' is OPEN")

        try:
            # Execute function
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)

            self._on_success()
            return result

        except Exception as exc:
            self._on_failure()
            raise exc

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test → back to OPEN
            self._transition_to_open()
        elif self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if timeout elapsed."""
        if not self.last_failure_time:
            return False

        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout_seconds

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass
```

## Integration with TriAgentSDLC

```python
# core/orchestrator/tri_agent_sdlc.py

from core.resilience.task_ledger import TaskLedger, TaskState
from core.resilience.retry_strategy import RetryStrategy, RetryConfig
from core.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

class TriAgentSDLCOrchestrator:

    def __init__(self, ...):
        # ... existing initialization ...

        # Task resilience (Issue #5)
        self.task_ledger = TaskLedger(
            db_path=Path("persistence/database/task_ledger.db")
        )

        self.retry_strategy = RetryStrategy(
            RetryConfig(max_retries=3, base_delay_ms=100)
        )

        # Circuit breakers per agent
        self.circuit_breakers = {
            "claude": CircuitBreaker("claude_code_agent"),
            "codex": CircuitBreaker("aider_codex_agent"),
            "gemini": CircuitBreaker("gemini_cli_agent"),
        }

    async def initialize(self):
        # ... existing initialization ...
        await self.task_ledger.initialize()

    async def _execute_agent_task_with_resilience(
        self,
        agent_name: str,
        task_type: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute agent task with full resilience:
        1. Enqueue in task ledger
        2. Execute through circuit breaker
        3. Retry with exponential backoff on transient failures
        4. Move to dead-letter queue on permanent failure
        """
        # Enqueue task
        task_id = await self.task_ledger.enqueue(
            agent_name=agent_name,
            task_type=task_type,
            payload=payload,
            priority=0,
            max_retries=3
        )

        try:
            # Mark as running
            await self.task_ledger.mark_running(task_id)

            # Get circuit breaker for agent
            breaker = self.circuit_breakers.get(agent_name)

            # Execute with retry + circuit breaker
            async def execute_task():
                # Execute through circuit breaker
                if breaker:
                    return await breaker.call(
                        self._execute_agent_task,
                        agent_name,
                        task_type,
                        payload
                    )
                else:
                    return await self._execute_agent_task(
                        agent_name,
                        task_type,
                        payload
                    )

            # Execute with exponential backoff retry
            result = await self.retry_strategy.execute_with_retry(execute_task)

            # Mark as done
            await self.task_ledger.mark_done(task_id, result)

            return result

        except CircuitBreakerOpen as exc:
            # Circuit breaker open → mark as blocked
            await self.task_ledger.mark_blocked(task_id, str(exc))
            raise

        except Exception as exc:
            # Permanent failure → mark as failed + dead-letter queue
            await self.task_ledger.mark_failed(task_id, str(exc))
            raise
```

## Acceptance Criteria

- [x] Task ledger with state transitions (TaskState enum + SQLite)
- [x] Retry logic with exponential backoff + jitter (RetryStrategy)
- [x] Circuit breaker per agent (CircuitBreaker with 5 failure threshold)
- [x] Dead-letter queue for permanently failed tasks (dead_letter_queue table)
- [ ] Unit tests for all state transitions
- [ ] Integration tests with TriAgentSDLC
- [ ] Documentation

## Testing Strategy

```python
# tests/test_resilience/test_task_ledger.py
- test_enqueue_task()
- test_state_transitions()
- test_retry_count_increment()
- test_dead_letter_queue()
- test_task_history()

# tests/test_resilience/test_retry_strategy.py
- test_exponential_backoff()
- test_jitter()
- test_max_delay_cap()
- test_retry_exhaustion()

# tests/test_resilience/test_circuit_breaker.py
- test_closed_to_open_transition()
- test_open_to_half_open_timeout()
- test_half_open_to_closed_recovery()
- test_half_open_to_open_on_failure()
- test_circuit_breaker_open_exception()
```

## Performance Targets

- Task state transition: <10ms (SQLite write)
- Circuit breaker overhead: <1ms
- Retry delay accuracy: ±10% (jitter)
- Dead-letter queue query: <100ms

## Security Considerations

- SQLite database permissions: 640 (owner read/write, group read)
- No sensitive data in task payloads (use references/IDs)
- Audit logging for circuit breaker trips
- Rate limiting on retry attempts (prevent infinite loops)

## Rollout Plan

1. Implement core components (TaskLedger, RetryStrategy, CircuitBreaker)
2. Write comprehensive unit tests
3. Integrate into TriAgentSDLC
4. Monitor metrics in production (circuit breaker trips, retry rates)
5. Tune parameters based on real-world failures

---

**Status**: Ready for implementation
**Next Steps**: Create `core/resilience/` directory and implement TaskLedger first
**ETA**: 3-5 days for full implementation + tests
