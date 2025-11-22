"""
Resilience Module - Task state management and fault tolerance.

Implements orchestration hardening for 24/7 continuous operation:
- Task Ledger: State machine with SQLite persistence
- Retry Strategy: Exponential backoff with jitter
- Circuit Breaker: Per-agent fault isolation

Components:
- TaskLedger: Task state management (queued → running → done/failed)
- RetryStrategy: Exponential backoff with configurable jitter
- CircuitBreaker: Circuit breaker pattern (closed → open → half-open)

Based on: Issue #5 - Task Ledger with retries, timeouts, circuit breakers
Implementation: Claude (Sonnet 4.5) + Codex + Gemini tri-agent collaboration
"""

from .task_ledger import (
    TaskLedger,
    Task,
    TaskState,
)

from .retry_strategy import (
    RetryStrategy,
    RetryConfig,
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpen,
)

__all__ = [
    # Task Ledger
    'TaskLedger',
    'Task',
    'TaskState',

    # Retry Strategy
    'RetryStrategy',
    'RetryConfig',

    # Circuit Breaker
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState',
    'CircuitBreakerOpen',
]
