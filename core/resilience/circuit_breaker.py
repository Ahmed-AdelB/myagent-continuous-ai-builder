"""
Circuit Breaker - Per-agent fault isolation.

Implements circuit breaker pattern to prevent cascade failures:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit tripped, rejecting all requests (fast-fail)
- HALF_OPEN: Testing recovery, allowing limited requests

State Transitions:
    CLOSED → OPEN: After failure_threshold consecutive failures
    OPEN → HALF_OPEN: After timeout_seconds elapsed
    HALF_OPEN → CLOSED: After success_threshold consecutive successes
    HALF_OPEN → OPEN: On any failure during recovery test

Benefits:
- Prevents cascade failures (stops calling failing service)
- Fast-fail instead of waiting for timeouts
- Automatic recovery testing
- Per-agent isolation (one agent failure doesn't affect others)

Performance Targets:
    - Circuit breaker overhead: <1ms per call
    - State check: O(1)
    - Thread-safe for concurrent calls

Based on: Issue #5 - Task Ledger with retries, timeouts, circuit breakers
Implementation: Claude (Sonnet 4.5)
"""

import asyncio
import logging
from enum import Enum
from typing import Callable, TypeVar, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Circuit tripped, rejecting requests
    HALF_OPEN = "half_open"    # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration.

    Attributes:
        failure_threshold: Consecutive failures before trip (default: 5)
        success_threshold: Consecutive successes to close from half-open (default: 2)
        timeout_seconds: Time before attempting half-open (default: 60s)
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open (fast-fail)."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern for agent fault tolerance.

    Prevents cascade failures by fast-failing when an agent is consistently
    failing, giving it time to recover.

    State Machine:
        CLOSED (normal):
            - Requests pass through
            - On success: reset failure counter
            - On failure: increment failure counter
            - If failure_count >= failure_threshold: transition to OPEN

        OPEN (circuit tripped):
            - All requests rejected with CircuitBreakerOpen
            - After timeout_seconds: transition to HALF_OPEN

        HALF_OPEN (recovery test):
            - Limited requests pass through
            - On success: increment success counter
            - If success_count >= success_threshold: transition to CLOSED
            - On failure: transition to OPEN

    Usage:
        # Create circuit breaker for agent
        breaker = CircuitBreaker("coder_agent")

        # Execute function through circuit breaker
        try:
            result = await breaker.call(agent.process_task, task)
        except CircuitBreakerOpen:
            logger.error("Circuit breaker open - agent unavailable")
            # Fallback logic here

    Thread Safety:
        This implementation is NOT thread-safe. Use asyncio.Lock if needed
        for concurrent access from multiple tasks.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name (e.g., "coder_agent", "claude", "codex")
            config: Circuit breaker configuration (uses defaults if None)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None

        # Statistics
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,  # Rejected due to circuit open
            "state_transitions": [],
        }

        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"success_threshold={self.config.success_threshold}, "
            f"timeout={self.config.timeout_seconds}s"
        )

    async def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function through circuit breaker.

        Args:
            fn: Function to execute (sync or async)
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Result from successful function execution

        Raises:
            CircuitBreakerOpen: If circuit is open (fast-fail)
            Original exception: If function fails

        Example:
            breaker = CircuitBreaker("api_service")

            async def call_api(param):
                return await api.fetch(param)

            try:
                result = await breaker.call(call_api, "value")
            except CircuitBreakerOpen:
                # Fallback logic
                result = get_cached_value("value")
        """
        self.stats["total_calls"] += 1

        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                # Circuit still open - reject request
                self.stats["rejected_calls"] += 1
                elapsed = (datetime.now() - self.last_failure_time).total_seconds() if self.last_failure_time else 0
                remaining = max(0, self.config.timeout_seconds - elapsed)

                logger.warning(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Rejecting request. Retry in {remaining:.1f}s"
                )

                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Agent unavailable. Retry in {remaining:.1f}s"
                )

        try:
            # Execute function
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)

            # Success!
            self._on_success()
            self.stats["successful_calls"] += 1

            return result

        except Exception as exc:
            # Failure
            self._on_failure()
            self.stats["failed_calls"] += 1

            logger.warning(
                f"Circuit breaker '{self.name}': Call failed with {type(exc).__name__}: {exc}"
            )

            # Re-raise original exception
            raise exc

    def _on_success(self) -> None:
        """Handle successful call."""
        # Reset failure count
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            # Increment success count during recovery test
            self.success_count += 1

            logger.debug(
                f"Circuit breaker '{self.name}' (HALF_OPEN): "
                f"Success {self.success_count}/{self.config.success_threshold}"
            )

            # Check if enough successes to close circuit
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test → back to OPEN
            logger.warning(
                f"Circuit breaker '{self.name}' (HALF_OPEN): "
                f"Recovery test failed. Transitioning to OPEN."
            )
            self._transition_to_open()

        elif self.state == CircuitState.CLOSED:
            # Check if threshold exceeded
            logger.debug(
                f"Circuit breaker '{self.name}' (CLOSED): "
                f"Failure {self.failure_count}/{self.config.failure_threshold}"
            )

            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """
        Check if timeout elapsed and circuit should attempt recovery.

        Returns:
            True if timeout elapsed, False otherwise
        """
        if not self.last_failure_time:
            return False

        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout_seconds

    def _transition_to_open(self) -> None:
        """Transition to OPEN state (circuit tripped)."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.success_count = 0

        self.stats["state_transitions"].append({
            "from": old_state.value,
            "to": CircuitState.OPEN.value,
            "timestamp": datetime.now().isoformat(),
            "reason": f"Failure threshold exceeded ({self.failure_count} failures)"
        })

        logger.error(
            f"⚠️ Circuit breaker '{self.name}' TRIPPED: "
            f"{old_state.value} → OPEN "
            f"({self.failure_count} consecutive failures). "
            f"Will attempt recovery in {self.config.timeout_seconds}s"
        )

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state (testing recovery)."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0

        self.stats["state_transitions"].append({
            "from": old_state.value,
            "to": CircuitState.HALF_OPEN.value,
            "timestamp": datetime.now().isoformat(),
            "reason": f"Timeout elapsed ({self.config.timeout_seconds}s)"
        })

        logger.info(
            f"Circuit breaker '{self.name}': {old_state.value} → HALF_OPEN. "
            f"Testing recovery..."
        )

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state (normal operation resumed)."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

        self.stats["state_transitions"].append({
            "from": old_state.value,
            "to": CircuitState.CLOSED.value,
            "timestamp": datetime.now().isoformat(),
            "reason": f"Recovery successful ({self.config.success_threshold} successes)"
        })

        logger.info(
            f"✅ Circuit breaker '{self.name}': {old_state.value} → CLOSED. "
            f"Normal operation resumed."
        )

    def reset(self) -> None:
        """
        Manually reset circuit breaker to CLOSED state.

        Use with caution - only when you know the underlying issue is resolved.
        """
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

        self.stats["state_transitions"].append({
            "from": old_state.value,
            "to": CircuitState.CLOSED.value,
            "timestamp": datetime.now().isoformat(),
            "reason": "Manual reset"
        })

        logger.warning(
            f"Circuit breaker '{self.name}' manually reset: {old_state.value} → CLOSED"
        )

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with call counts, state transitions, etc.
        """
        return {
            **self.stats,
            "current_state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
            }
        }

    def is_available(self) -> bool:
        """
        Check if circuit breaker is available to handle requests.

        Returns:
            True if CLOSED or HALF_OPEN, False if OPEN
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self._should_attempt_reset():
                return True  # Will transition to HALF_OPEN on next call
            return False

        return True  # CLOSED or HALF_OPEN


class CircuitBreakerManager:
    """
    Manage multiple circuit breakers.

    Provides centralized access to circuit breakers for different agents/services.

    Usage:
        manager = CircuitBreakerManager()
        manager.register("coder_agent", CircuitBreakerConfig(failure_threshold=5))
        manager.register("gemini", CircuitBreakerConfig(failure_threshold=3))

        # Execute through circuit breaker
        result = await manager.call("coder_agent", agent.process_task, task)

        # Get all stats
        all_stats = manager.get_all_stats()
    """

    def __init__(self):
        """Initialize circuit breaker manager."""
        self.breakers: dict[str, CircuitBreaker] = {}

        logger.info("CircuitBreakerManager initialized")

    def register(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Register a new circuit breaker.

        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration

        Returns:
            CircuitBreaker instance
        """
        if name in self.breakers:
            logger.warning(f"Circuit breaker '{name}' already registered. Overwriting.")

        breaker = CircuitBreaker(name, config)
        self.breakers[name] = breaker

        logger.info(f"Registered circuit breaker: {name}")

        return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.breakers.get(name)

    async def call(
        self,
        name: str,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """
        Execute function through named circuit breaker.

        Args:
            name: Circuit breaker name
            fn: Function to execute
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            ValueError: If circuit breaker not found
            CircuitBreakerOpen: If circuit is open
        """
        breaker = self.breakers.get(name)
        if not breaker:
            raise ValueError(f"Circuit breaker '{name}' not found. Register it first.")

        return await breaker.call(fn, *args, **kwargs)

    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        for name, breaker in self.breakers.items():
            breaker.reset()
            logger.info(f"Reset circuit breaker: {name}")
