"""In-memory, thread-safe sliding-window rate limiter for FastAPI endpoints.

Provides:
- InMemoryRateLimiter: Thread-safe sliding-window rate limiter using standard-library primitives.
- Deterministic clock injection for automated testing without sleeping.
- Bounded memory usage with automatic pruning and conservative rejection safeguards.
- Dedicated singleton dependencies for authentication endpoints (login and registration).
"""

from collections import deque
import math
import threading
import time
from typing import Callable, Deque, Dict, Optional
from fastapi import HTTPException, Request, status

from app.core.config import settings


class InMemoryRateLimiter:
    """Thread-safe sliding-window in-memory rate limiter using standard library components."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        namespace: str = "default",
        max_keys: int = 10000,
        clock_fn: Optional[Callable[[], float]] = None,
    ):
        if max_requests <= 0:
            raise ValueError("max_requests must be a positive integer.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be a positive integer.")
        if max_keys <= 0:
            raise ValueError("max_keys must be a positive integer.")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.namespace = namespace
        self.max_keys = max_keys
        self.default_clock = time.monotonic
        self.clock: Callable[[], float] = clock_fn or self.default_clock

        # Storage mapping: key -> deque of timestamps
        self._records: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def get_client_ip(self, request: Request) -> str:
        """Extract socket peer IP from request.client.

        IMPORTANT:
        Strictly ignores arbitrary 'X-Forwarded-For' and 'X-Real-IP' headers to prevent
        header spoofing attacks. Always reads directly from the TCP socket client peer name.
        """
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    def _cleanup_expired_locked(self, cutoff: float) -> int:
        """Purge empty or expired queues from memory. Must be called with self._lock held."""
        stale_keys = []
        for key, queue in self._records.items():
            while queue and queue[0] <= cutoff:
                queue.popleft()
            if not queue:
                stale_keys.append(key)

        for key in stale_keys:
            del self._records[key]

        return len(stale_keys)

    def check_rate_limit(self, request: Request) -> None:
        """Evaluate rate limit for the incoming request socket IP.

        Raises:
            HTTPException: 429 Too Many Requests with integer Retry-After header
                           if quota is exceeded or maximum-key capacity is saturated.
        """
        client_ip = self.get_client_ip(request)
        key = f"{self.namespace}:{client_ip}"
        now = self.clock()
        cutoff = now - self.window_seconds

        with self._lock:
            if key in self._records:
                queue = self._records[key]
                # Prune timestamps outside the active sliding window
                while queue and queue[0] <= cutoff:
                    queue.popleft()
            else:
                # Key is new. Check capacity and memory bounds
                if len(self._records) >= self.max_keys:
                    self._cleanup_expired_locked(cutoff)
                    if len(self._records) >= self.max_keys:
                        # Safeguard: memory limit reached and no stale entries could be purged.
                        # Reject conservatively rather than bypassing the rate limiter.
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Too many requests. Please try again later.",
                            headers={"Retry-After": str(self.window_seconds)},
                        )

                queue = deque()
                self._records[key] = queue

            if len(queue) >= self.max_requests:
                oldest_timestamp = queue[0]
                remaining_seconds = self.window_seconds - (now - oldest_timestamp)
                retry_after = max(1, math.ceil(remaining_seconds))

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Record this attempt
            queue.append(now)

    def __call__(self, request: Request) -> None:
        """FastAPI route dependency hook."""
        self.check_rate_limit(request)

    def cleanup(self) -> int:
        """Public thread-safe manual cleanup of expired entries."""
        now = self.clock()
        cutoff = now - self.window_seconds
        with self._lock:
            return self._cleanup_expired_locked(cutoff)

    def reset(self) -> None:
        """Reset limiter records and restore default monotonic clock for test isolation."""
        with self._lock:
            self._records.clear()
            self.clock = self.default_clock


# Pre-configured singleton instances for authentication endpoints
login_rate_limiter = InMemoryRateLimiter(
    max_requests=settings.RATE_LIMIT_LOGIN_MAX_ATTEMPTS,
    window_seconds=settings.RATE_LIMIT_LOGIN_WINDOW_SECONDS,
    namespace="auth:login",
    max_keys=settings.RATE_LIMIT_MAX_KEYS,
)

register_rate_limiter = InMemoryRateLimiter(
    max_requests=settings.RATE_LIMIT_REGISTER_MAX_ATTEMPTS,
    window_seconds=settings.RATE_LIMIT_REGISTER_WINDOW_SECONDS,
    namespace="auth:register",
    max_keys=settings.RATE_LIMIT_MAX_KEYS,
)

memory_rate_limiter = InMemoryRateLimiter(
    max_requests=getattr(settings, "RATE_LIMIT_MEMORY_MAX_REQUESTS", 60),
    window_seconds=getattr(settings, "RATE_LIMIT_MEMORY_WINDOW_SECONDS", 60),
    namespace="memory",
    max_keys=settings.RATE_LIMIT_MAX_KEYS,
)

cross_meeting_rate_limiter = InMemoryRateLimiter(
    max_requests=getattr(settings, "RATE_LIMIT_CROSS_MEETING_MAX_REQUESTS", 120),
    window_seconds=getattr(settings, "RATE_LIMIT_CROSS_MEETING_WINDOW_SECONDS", 60),
    namespace="cross_meeting",
    max_keys=settings.RATE_LIMIT_MAX_KEYS,
)
