"""
Unit tests for CircuitBreaker.

Tests:
- CLOSED → OPEN transition (trip on failures)
- OPEN → HALF_OPEN transition (timeout)
- HALF_OPEN → CLOSED transition (recovery)
- HALF_OPEN → OPEN transition (failure during recovery)
- CircuitBreakerOpen exception
- Statistics collection

Issue #5: Task Ledger with retries, timeouts, circuit breakers
"""

import pytest
import asyncio
from unittest.mock import AsyncMock
from datetime import datetime

from core.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpen,
    CircuitBreakerManager
)


@pytest.fixture
def circuit_breaker():
    """Create circuit breaker with test config."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=1  # Short timeout for tests
    )
    return CircuitBreaker("test_service", config)


@pytest.mark.asyncio
async def test_initial_state_closed(circuit_breaker):
    """Test circuit breaker starts in CLOSED state."""
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0
    assert circuit_breaker.success_count == 0


@pytest.mark.asyncio
async def test_successful_call_in_closed_state(circuit_breaker):
    """Test successful call in CLOSED state."""
    async def successful_fn():
        return "success"

    result = await circuit_breaker.call(successful_fn)

    assert result == "success"
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_failed_call_increments_failure_count(circuit_breaker):
    """Test failed call increments failure count."""
    async def failing_fn():
        raise ValueError("Error")

    with pytest.raises(ValueError):
        await circuit_breaker.call(failing_fn)

    assert circuit_breaker.failure_count == 1
    assert circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_transition_closed_to_open(circuit_breaker):
    """Test CLOSED → OPEN transition after threshold failures."""
    async def failing_fn():
        raise ValueError("Error")

    # Trigger 3 failures (threshold = 3)
    for i in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_fn)

    # Circuit should trip to OPEN
    assert circuit_breaker.state == CircuitState.OPEN
    assert circuit_breaker.failure_count == 3


@pytest.mark.asyncio
async def test_circuit_breaker_open_rejects_requests(circuit_breaker):
    """Test OPEN circuit breaker rejects requests."""
    async def failing_fn():
        raise ValueError("Error")

    # Trip circuit
    for i in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_fn)

    # Circuit is OPEN - should reject with CircuitBreakerOpen
    async def any_fn():
        return "should not execute"

    with pytest.raises(CircuitBreakerOpen, match="Circuit breaker 'test_service' is OPEN"):
        await circuit_breaker.call(any_fn)

    # Verify stats
    stats = circuit_breaker.get_stats()
    assert stats["rejected_calls"] == 1


@pytest.mark.asyncio
async def test_transition_open_to_half_open(circuit_breaker):
    """Test OPEN → HALF_OPEN transition after timeout."""
    async def failing_fn():
        raise ValueError("Error")

    # Trip circuit to OPEN
    for i in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_fn)

    assert circuit_breaker.state == CircuitState.OPEN

    # Wait for timeout (1 second)
    await asyncio.sleep(1.1)

    # Next call should transition to HALF_OPEN
    async def test_fn():
        return "testing recovery"

    result = await circuit_breaker.call(test_fn)

    assert result == "testing recovery"
    assert circuit_breaker.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_transition_half_open_to_closed(circuit_breaker):
    """Test HALF_OPEN → CLOSED transition after success threshold."""
    async def failing_fn():
        raise ValueError("Error")

    # Trip circuit
    for i in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_fn)

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Transition to HALF_OPEN with successful calls
    async def successful_fn():
        return "success"

    # Need 2 successful calls (success_threshold = 2)
    result1 = await circuit_breaker.call(successful_fn)
    assert circuit_breaker.state == CircuitState.HALF_OPEN
    assert circuit_breaker.success_count == 1

    result2 = await circuit_breaker.call(successful_fn)
    assert circuit_breaker.state == CircuitState.CLOSED  # Should close after 2 successes
    assert circuit_breaker.success_count == 0  # Reset
    assert circuit_breaker.failure_count == 0  # Reset


@pytest.mark.asyncio
async def test_transition_half_open_to_open_on_failure(circuit_breaker):
    """Test HALF_OPEN → OPEN transition on failure during recovery."""
    async def failing_fn():
        raise ValueError("Error")

    # Trip circuit
    for i in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_fn)

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Successful call to enter HALF_OPEN
    async def successful_fn():
        return "success"

    await circuit_breaker.call(successful_fn)
    assert circuit_breaker.state == CircuitState.HALF_OPEN

    # Failure during recovery → back to OPEN
    with pytest.raises(ValueError):
        await circuit_breaker.call(failing_fn)

    assert circuit_breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_successful_call_resets_failure_count(circuit_breaker):
    """Test successful call resets failure count."""
    async def failing_fn():
        raise ValueError("Error")

    async def successful_fn():
        return "success"

    # 2 failures
    for i in range(2):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_fn)

    assert circuit_breaker.failure_count == 2

    # Successful call should reset
    await circuit_breaker.call(successful_fn)
    assert circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_manual_reset(circuit_breaker):
    """Test manual reset to CLOSED state."""
    async def failing_fn():
        raise ValueError("Error")

    # Trip circuit
    for i in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_fn)

    assert circuit_breaker.state == CircuitState.OPEN

    # Manual reset
    circuit_breaker.reset()

    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0
    assert circuit_breaker.success_count == 0


@pytest.mark.asyncio
async def test_is_available():
    """Test is_available() method."""
    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1))

    async def failing_fn():
        raise ValueError("Error")

    # Initially available
    assert breaker.is_available() is True

    # Trip circuit
    for i in range(2):
        with pytest.raises(ValueError):
            await breaker.call(failing_fn)

    # OPEN - not available
    assert breaker.is_available() is False

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Should be available again (will transition to HALF_OPEN)
    assert breaker.is_available() is True


@pytest.mark.asyncio
async def test_statistics():
    """Test statistics collection."""
    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))

    async def successful_fn():
        return "success"

    async def failing_fn():
        raise ValueError("Error")

    # 2 successful calls
    await breaker.call(successful_fn)
    await breaker.call(successful_fn)

    # 2 failed calls (trips circuit)
    for i in range(2):
        with pytest.raises(ValueError):
            await breaker.call(failing_fn)

    # 1 rejected call (circuit open)
    with pytest.raises(CircuitBreakerOpen):
        await breaker.call(successful_fn)

    stats = breaker.get_stats()

    assert stats["total_calls"] == 5  # 2 success + 2 fail + 1 reject
    assert stats["successful_calls"] == 2
    assert stats["failed_calls"] == 2
    assert stats["rejected_calls"] == 1
    assert stats["current_state"] == CircuitState.OPEN.value
    assert len(stats["state_transitions"]) == 1  # CLOSED → OPEN


@pytest.mark.asyncio
async def test_sync_callable():
    """Test circuit breaker with sync callable."""
    breaker = CircuitBreaker("test")

    def sync_fn():
        return "sync_result"

    result = await breaker.call(sync_fn)
    assert result == "sync_result"


@pytest.mark.asyncio
async def test_circuit_breaker_manager():
    """Test CircuitBreakerManager."""
    manager = CircuitBreakerManager()

    # Register circuit breakers
    config = CircuitBreakerConfig(failure_threshold=2)
    manager.register("service_a", config)
    manager.register("service_b", config)

    # Get circuit breaker
    breaker_a = manager.get("service_a")
    assert breaker_a is not None
    assert breaker_a.name == "service_a"

    # Execute through manager
    async def successful_fn():
        return "success"

    result = await manager.call("service_a", successful_fn)
    assert result == "success"

    # Get all stats
    all_stats = manager.get_all_stats()
    assert "service_a" in all_stats
    assert "service_b" in all_stats


@pytest.mark.asyncio
async def test_circuit_breaker_manager_not_found():
    """Test calling unknown circuit breaker raises ValueError."""
    manager = CircuitBreakerManager()

    async def fn():
        return "test"

    with pytest.raises(ValueError, match="Circuit breaker 'unknown' not found"):
        await manager.call("unknown", fn)


@pytest.mark.asyncio
async def test_circuit_breaker_manager_reset_all():
    """Test resetting all circuit breakers."""
    manager = CircuitBreakerManager()

    config = CircuitBreakerConfig(failure_threshold=2)
    manager.register("service_a", config)
    manager.register("service_b", config)

    # Trip both circuits
    async def failing_fn():
        raise ValueError("Error")

    for i in range(2):
        with pytest.raises(ValueError):
            await manager.call("service_a", failing_fn)
        with pytest.raises(ValueError):
            await manager.call("service_b", failing_fn)

    # Both should be OPEN
    assert manager.get("service_a").state == CircuitState.OPEN
    assert manager.get("service_b").state == CircuitState.OPEN

    # Reset all
    manager.reset_all()

    # Both should be CLOSED
    assert manager.get("service_a").state == CircuitState.CLOSED
    assert manager.get("service_b").state == CircuitState.CLOSED
