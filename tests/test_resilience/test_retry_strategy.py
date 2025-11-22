"""
Unit tests for RetryStrategy.

Tests:
- Exponential backoff calculation
- Jitter randomness
- Max delay cap
- Retry exhaustion
- Success on retry
- Custom retry conditions

Issue #5: Task Ledger with retries, timeouts, circuit breakers
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.resilience.retry_strategy import RetryStrategy, RetryConfig


def test_calculate_delay_exponential_backoff():
    """Test exponential backoff calculation."""
    config = RetryConfig(
        base_delay_ms=100,
        exponential_base=2.0,
        jitter=False  # Disable jitter for deterministic test
    )
    strategy = RetryStrategy(config)

    # Attempt 0: 100ms
    delay_0 = strategy.calculate_delay(0)
    assert 0.09 < delay_0 < 0.11  # ~0.1s

    # Attempt 1: 200ms
    delay_1 = strategy.calculate_delay(1)
    assert 0.19 < delay_1 < 0.21  # ~0.2s

    # Attempt 2: 400ms
    delay_2 = strategy.calculate_delay(2)
    assert 0.39 < delay_2 < 0.41  # ~0.4s

    # Attempt 3: 800ms
    delay_3 = strategy.calculate_delay(3)
    assert 0.79 < delay_3 < 0.81  # ~0.8s


def test_calculate_delay_with_jitter():
    """Test jitter adds ±50% randomness."""
    config = RetryConfig(
        base_delay_ms=100,
        exponential_base=2.0,
        jitter=True
    )
    strategy = RetryStrategy(config)

    # Attempt 0: 100ms * jitter(0.5-1.5) = 50-150ms
    delay = strategy.calculate_delay(0)
    assert 0.05 <= delay <= 0.15  # 50-150ms

    # Run multiple times to verify randomness
    delays = [strategy.calculate_delay(0) for _ in range(10)]
    assert len(set(delays)) > 1  # Should have different values


def test_calculate_delay_max_cap():
    """Test max delay cap is enforced."""
    config = RetryConfig(
        base_delay_ms=100,
        exponential_base=2.0,
        max_delay_ms=500,  # Cap at 500ms
        jitter=False
    )
    strategy = RetryStrategy(config)

    # Attempt 10: 100 * 2^10 = 102,400ms → capped at 500ms
    delay = strategy.calculate_delay(10)
    assert delay <= 0.5  # Should not exceed 500ms


@pytest.mark.asyncio
async def test_execute_with_retry_success_first_attempt():
    """Test successful execution on first attempt."""
    async def successful_fn():
        return "success"

    strategy = RetryStrategy(RetryConfig(max_retries=3))
    result = await strategy.execute_with_retry(successful_fn)

    assert result == "success"


@pytest.mark.asyncio
async def test_execute_with_retry_success_on_third_attempt():
    """Test successful execution after 2 failures."""
    call_count = 0

    async def flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception(f"Transient error {call_count}")
        return "success"

    config = RetryConfig(
        max_retries=3,
        base_delay_ms=10,  # Fast retries for test
        jitter=False
    )
    strategy = RetryStrategy(config)

    result = await strategy.execute_with_retry(flaky_fn)

    assert result == "success"
    assert call_count == 3  # Failed twice, succeeded on third


@pytest.mark.asyncio
async def test_execute_with_retry_exhaustion():
    """Test retry exhaustion raises last exception."""
    call_count = 0

    async def always_fail_fn():
        nonlocal call_count
        call_count += 1
        raise ValueError(f"Permanent error {call_count}")

    config = RetryConfig(
        max_retries=2,  # Allow 2 retries = 3 total attempts
        base_delay_ms=10,
        jitter=False
    )
    strategy = RetryStrategy(config)

    with pytest.raises(ValueError, match="Permanent error 3"):
        await strategy.execute_with_retry(always_fail_fn)

    assert call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_execute_with_retry_delay_increases():
    """Test that delay increases exponentially between retries."""
    import time

    call_times = []

    async def failing_fn():
        call_times.append(time.time())
        raise Exception("Error")

    config = RetryConfig(
        max_retries=2,
        base_delay_ms=100,  # 100ms, 200ms
        jitter=False
    )
    strategy = RetryStrategy(config)

    with pytest.raises(Exception):
        await strategy.execute_with_retry(failing_fn)

    # Verify delays between calls
    assert len(call_times) == 3

    delay_1 = call_times[1] - call_times[0]
    delay_2 = call_times[2] - call_times[1]

    # First delay: ~100ms
    assert 0.08 < delay_1 < 0.15

    # Second delay: ~200ms (should be ~2x first delay)
    assert 0.15 < delay_2 < 0.30


@pytest.mark.asyncio
async def test_execute_with_custom_retry_condition():
    """Test custom retry condition."""
    call_count = 0

    async def fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("Transient timeout")
        elif call_count == 2:
            raise ValueError("Permanent error")
        return "success"

    def is_transient_error(exc: Exception) -> bool:
        """Only retry TimeoutError."""
        return isinstance(exc, TimeoutError)

    config = RetryConfig(max_retries=3, base_delay_ms=10)
    strategy = RetryStrategy(config)

    # Should retry on TimeoutError, then fail on ValueError
    with pytest.raises(ValueError, match="Permanent error"):
        await strategy.execute_with_custom_retry_condition(
            fn,
            is_transient_error
        )

    assert call_count == 2  # First attempt + one retry


def test_get_total_delay():
    """Test total delay calculation."""
    config = RetryConfig(
        max_retries=3,
        base_delay_ms=100,
        exponential_base=2.0,
        jitter=False  # No jitter for deterministic calculation
    )
    strategy = RetryStrategy(config)

    # Total: 100ms + 200ms + 400ms = 700ms
    total_delay = strategy.get_total_delay()
    assert 0.69 < total_delay < 0.71  # ~0.7s


def test_sync_retry():
    """Test synchronous retry."""
    call_count = 0

    def sync_fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Error")
        return "success"

    config = RetryConfig(max_retries=3, base_delay_ms=10)
    strategy = RetryStrategy(config)

    result = strategy.execute_with_retry_sync(sync_fn)

    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_zero_retries():
    """Test with zero retries (fail immediately)."""
    call_count = 0

    async def failing_fn():
        nonlocal call_count
        call_count += 1
        raise Exception("Error")

    config = RetryConfig(max_retries=0)  # No retries
    strategy = RetryStrategy(config)

    with pytest.raises(Exception):
        await strategy.execute_with_retry(failing_fn)

    assert call_count == 1  # Only one attempt


@pytest.mark.asyncio
async def test_async_and_sync_callable():
    """Test handling both async and sync callables."""
    async def async_fn():
        return "async_result"

    def sync_fn():
        return "sync_result"

    strategy = RetryStrategy()

    async_result = await strategy.execute_with_retry(async_fn)
    assert async_result == "async_result"

    sync_result = await strategy.execute_with_retry(sync_fn)
    assert sync_result == "sync_result"
