"""
Retry Handler with Exponential Backoff

Provides resilient LLM API calls with automatic retry on transient failures.
Prevents rate limits, timeouts, and network issues from stopping continuous execution.

Based on research showing exponential backoff is essential for production AI systems.
"""

import asyncio
import random
import time
from typing import Callable, TypeVar, Optional, Type, Tuple
from functools import wraps
from enum import Enum
from loguru import logger

T = TypeVar('T')


class ErrorClassification(Enum):
    """Classification of errors for retry logic"""
    RETRIABLE = "retriable"  # Retry with backoff
    NON_RETRIABLE = "non_retriable"  # Fail fast
    RATE_LIMIT = "rate_limit"  # Longer backoff


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        rate_limit_delay: float = 120.0  # 2 minutes for rate limits
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.rate_limit_delay = rate_limit_delay


class MaxRetriesExceeded(Exception):
    """Raised when maximum retry attempts are exhausted"""
    pass


def classify_error(error: Exception) -> ErrorClassification:
    """
    Classify error to determine retry strategy.

    Args:
        error: The exception that occurred

    Returns:
        ErrorClassification for retry decision
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Rate limit errors - need longer backoff
    rate_limit_indicators = [
        'rate limit',
        'too many requests',
        '429',
        'quota exceeded',
        'ratelimit'
    ]
    if any(indicator in error_str for indicator in rate_limit_indicators):
        return ErrorClassification.RATE_LIMIT

    # Retriable errors - transient issues
    retriable_types = (
        ConnectionError,
        TimeoutError,
        OSError,  # Network issues
        asyncio.TimeoutError
    )
    if isinstance(error, retriable_types):
        return ErrorClassification.RETRIABLE

    retriable_keywords = [
        'timeout',
        'connection',
        'temporary',
        'unavailable',
        'overloaded',
        'server error',
        '500',
        '502',
        '503',
        '504'
    ]
    if any(keyword in error_str for keyword in retriable_keywords):
        return ErrorClassification.RETRIABLE

    # Non-retriable errors - fail fast
    non_retriable_keywords = [
        'invalid',
        'unauthorized',
        '401',
        '403',
        'forbidden',
        'not found',
        '404',
        'bad request',
        '400',
        'authentication'
    ]
    if any(keyword in error_str for keyword in non_retriable_keywords):
        return ErrorClassification.NON_RETRIABLE

    # Default to retriable for unknown errors (conservative approach)
    return ErrorClassification.RETRIABLE


async def retry_with_exponential_backoff(
    func: Callable,
    config: RetryConfig = None,
    error_classifier: Callable[[Exception], ErrorClassification] = None,
    *args,
    **kwargs
) -> T:
    """
    Execute function with exponential backoff retry.

    Args:
        func: Async function to execute
        config: Retry configuration
        error_classifier: Optional custom error classifier
        *args, **kwargs: Arguments to pass to func

    Returns:
        Result from successful function execution

    Raises:
        MaxRetriesExceeded: If all retry attempts exhausted
        Exception: If non-retriable error encountered
    """
    config = config or RetryConfig()
    error_classifier = error_classifier or classify_error

    attempt = 0
    last_error = None

    while attempt < config.max_attempts:
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            last_error = e
            classification = error_classifier(e)

            # Non-retriable error - fail immediately
            if classification == ErrorClassification.NON_RETRIABLE:
                logger.error(f"Non-retriable error encountered: {e}")
                raise

            attempt += 1

            # All retries exhausted
            if attempt >= config.max_attempts:
                logger.error(
                    f"Max retries ({config.max_attempts}) exhausted for {func.__name__}"
                )
                break

            # Calculate delay
            if classification == ErrorClassification.RATE_LIMIT:
                delay = config.rate_limit_delay
                logger.warning(
                    f"Rate limit hit on attempt {attempt}. "
                    f"Waiting {delay}s before retry."
                )
            else:
                # Exponential backoff
                delay = min(
                    config.base_delay * (config.exponential_base ** (attempt - 1)),
                    config.max_delay
                )

            # Add jitter to prevent thundering herd
            if config.jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed: {e}. "
                f"Retrying in {delay:.2f}s"
            )

            await asyncio.sleep(delay)

    # All retries exhausted
    raise MaxRetriesExceeded(
        f"Failed after {attempt} attempts. Last error: {last_error}"
    ) from last_error


def with_retry(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """
    Decorator for automatic retry with exponential backoff.

    Usage:
        @with_retry(max_attempts=3)
        async def call_llm_api():
            return await llm.generate(...)

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay cap

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay
            )
            return await retry_with_exponential_backoff(
                func,
                config,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


# Example usage for LangChain LLM calls
@with_retry(max_attempts=5, base_delay=2.0)
async def call_llm_with_retry(llm, prompt: str, **kwargs):
    """
    Call LLM with automatic retry on failures.

    This is a helper function that wraps LangChain LLM calls.
    """
    if hasattr(llm, 'ainvoke'):
        return await llm.ainvoke(prompt, **kwargs)
    elif hasattr(llm, 'agenerate'):
        result = await llm.agenerate([prompt], **kwargs)
        return result.generations[0][0].text
    else:
        # Fallback to synchronous call in thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, llm, prompt)
