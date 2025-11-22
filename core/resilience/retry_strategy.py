"""
Retry Strategy - Exponential backoff with jitter.

Implements retry logic for transient failures with intelligent backoff:
- Exponential backoff: delay = base * (exponential_base ** attempt)
- Jitter: Randomized delay to prevent thundering herd
- Configurable max delay cap
- Support for both sync and async functions

Backoff Formula:
    delay = min(base_delay * (exponential_base ** attempt), max_delay)
    if jitter: delay = delay * random(0.5, 1.5)

Example (base_delay=100ms, exponential_base=2, jitter=True):
    Attempt 1: 100ms ± 50% = 50-150ms
    Attempt 2: 200ms ± 50% = 100-300ms
    Attempt 3: 400ms ± 50% = 200-600ms
    Attempt 4: 800ms ± 50% = 400-1200ms

Performance Targets:
    - Delay calculation: <1ms
    - Jitter randomness: Uniform distribution
    - No thundering herd on retry storms

Based on: Issue #5 - Task Ledger with retries, timeouts, circuit breakers
Implementation: Claude (Sonnet 4.5)
"""

import asyncio
import random
import logging
from typing import Callable, TypeVar, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """
    Retry strategy configuration.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay_ms: Initial delay in milliseconds (default: 100ms)
        max_delay_ms: Maximum delay cap in milliseconds (default: 30s)
        exponential_base: Base for exponential backoff (default: 2.0)
        jitter: Add randomness to prevent thundering herd (default: True)
    """
    max_retries: int = 3
    base_delay_ms: float = 100.0
    max_delay_ms: float = 30000.0  # 30 seconds
    exponential_base: float = 2.0
    jitter: bool = True


class RetryStrategy:
    """
    Exponential backoff with jitter.

    Features:
    - Exponential backoff with configurable base
    - Jitter to prevent thundering herd (±50% randomness)
    - Max delay cap to prevent excessive waits
    - Support for both sync and async callables

    Usage:
        # Basic usage
        strategy = RetryStrategy()
        result = await strategy.execute_with_retry(risky_function, arg1, arg2)

        # Custom configuration
        config = RetryConfig(max_retries=5, base_delay_ms=200, jitter=True)
        strategy = RetryStrategy(config)
        result = await strategy.execute_with_retry(api_call)

        # Manual delay calculation
        delay_seconds = strategy.calculate_delay(attempt=2)
        await asyncio.sleep(delay_seconds)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry strategy.

        Args:
            config: Retry configuration (uses defaults if None)
        """
        self.config = config or RetryConfig()

        logger.debug(
            f"RetryStrategy initialized: max_retries={self.config.max_retries}, "
            f"base_delay={self.config.base_delay_ms}ms, "
            f"exponential_base={self.config.exponential_base}, "
            f"jitter={self.config.jitter}"
        )

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.

        Formula:
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            if jitter: delay = delay * random(0.5, 1.5)

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds

        Example:
            >>> strategy = RetryStrategy(RetryConfig(base_delay_ms=100, exponential_base=2))
            >>> strategy.calculate_delay(0)  # First retry
            0.1  # 100ms
            >>> strategy.calculate_delay(1)  # Second retry
            0.2  # 200ms
            >>> strategy.calculate_delay(2)  # Third retry
            0.4  # 400ms
        """
        # Calculate exponential delay
        delay_ms = self.config.base_delay_ms * (self.config.exponential_base ** attempt)

        # Apply max delay cap
        delay_ms = min(delay_ms, self.config.max_delay_ms)

        # Add jitter (±50%)
        if self.config.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay_ms *= jitter_factor

        # Convert to seconds
        delay_seconds = delay_ms / 1000.0

        logger.debug(
            f"Retry attempt {attempt}: delay={delay_ms:.1f}ms ({delay_seconds:.3f}s)"
        )

        return delay_seconds

    async def execute_with_retry(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """
        Execute function with exponential backoff retry.

        Retries on any exception. If all retries are exhausted, raises the last exception.

        Args:
            fn: Function to execute (sync or async)
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Result from successful function execution

        Raises:
            Last exception if all retries exhausted

        Example:
            async def unstable_api_call(param):
                # May raise TransientError
                return await api.fetch(param)

            strategy = RetryStrategy(RetryConfig(max_retries=3))
            result = await strategy.execute_with_retry(unstable_api_call, "value")
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Execute function (handle both sync and async)
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)

                # Success!
                if attempt > 0:
                    logger.info(f"✓ Retry succeeded on attempt {attempt + 1}")

                return result

            except Exception as exc:
                last_exception = exc

                if attempt < self.config.max_retries:
                    # Calculate delay and retry
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed: {exc}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded
                    logger.error(
                        f"All {self.config.max_retries + 1} attempts failed. "
                        f"Last error: {exc}"
                    )
                    raise last_exception

        # Should never reach here
        raise last_exception  # type: ignore

    def execute_with_retry_sync(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """
        Synchronous version of execute_with_retry.

        Use this for non-async functions. Note: Still uses asyncio.sleep internally,
        so this must be called from an async context.

        Args:
            fn: Synchronous function to execute
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Result from successful function execution

        Raises:
            Last exception if all retries exhausted
        """
        import time

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = fn(*args, **kwargs)

                if attempt > 0:
                    logger.info(f"✓ Retry succeeded on attempt {attempt + 1}")

                return result

            except Exception as exc:
                last_exception = exc

                if attempt < self.config.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed: {exc}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_retries + 1} attempts failed. "
                        f"Last error: {exc}"
                    )
                    raise last_exception

        raise last_exception  # type: ignore

    async def execute_with_custom_retry_condition(
        self,
        fn: Callable[..., T],
        should_retry: Callable[[Exception], bool],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """
        Execute with custom retry condition.

        Only retries if should_retry(exception) returns True.
        Useful for retrying only on transient errors (e.g., network timeouts).

        Args:
            fn: Function to execute
            should_retry: Predicate function to determine if exception should trigger retry
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Result from successful function execution

        Raises:
            Exception if should_retry returns False OR all retries exhausted

        Example:
            def is_transient_error(exc: Exception) -> bool:
                return isinstance(exc, (TimeoutError, ConnectionError))

            result = await strategy.execute_with_custom_retry_condition(
                api_call,
                is_transient_error,
                param1,
                param2
            )
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)

                if attempt > 0:
                    logger.info(f"✓ Retry succeeded on attempt {attempt + 1}")

                return result

            except Exception as exc:
                last_exception = exc

                # Check if should retry
                if not should_retry(exc):
                    logger.warning(f"Exception not retryable: {exc}")
                    raise exc

                if attempt < self.config.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed: {exc}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_retries + 1} attempts failed. "
                        f"Last error: {exc}"
                    )
                    raise last_exception

        raise last_exception  # type: ignore

    def get_total_delay(self) -> float:
        """
        Calculate total delay across all retry attempts (without jitter).

        Returns:
            Total delay in seconds

        Example:
            >>> strategy = RetryStrategy(RetryConfig(max_retries=3, base_delay_ms=100))
            >>> strategy.get_total_delay()
            0.7  # 100ms + 200ms + 400ms = 700ms
        """
        total_delay_ms = 0.0

        for attempt in range(self.config.max_retries):
            delay_ms = self.config.base_delay_ms * (self.config.exponential_base ** attempt)
            delay_ms = min(delay_ms, self.config.max_delay_ms)
            total_delay_ms += delay_ms

        return total_delay_ms / 1000.0
