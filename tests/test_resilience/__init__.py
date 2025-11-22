"""
Unit tests for resilience module.

Tests:
- task_ledger.py: State machine, transitions, dead-letter queue
- retry_strategy.py: Exponential backoff, jitter, retry exhaustion
- circuit_breaker.py: State transitions, circuit trips, recovery

Issue #5: Task Ledger with retries, timeouts, circuit breakers
"""
