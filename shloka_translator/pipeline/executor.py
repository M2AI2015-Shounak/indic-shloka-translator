"""Safe parallel execution, retries with backoff, and a token-bucket rate limiter."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class RateLimiter:
    """Thread-safe token bucket limiting requests per minute.

    Example:
        >>> rl = RateLimiter(6000)
        >>> rl.acquire()  # returns immediately while tokens are available
    """

    def __init__(self, requests_per_minute: float, burst: int | None = None) -> None:
        """Create the bucket.

        Args:
            requests_per_minute: Sustained rate; ``<= 0`` disables limiting.
            burst: Bucket capacity (defaults to ``max(1, rpm / 10)``).
        """
        self.rate = max(0.0, requests_per_minute) / 60.0
        self.capacity = float(burst or max(1, int(requests_per_minute / 10)))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one request token is available."""
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(min(wait, 1.0))


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter.

    Attributes:
        max_retries: Retries after the first attempt.
        base_delay: First backoff delay (seconds).
        max_delay: Cap for one delay (seconds).
    """

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Compute the delay before retry number ``attempt`` (1-based).

        Args:
            attempt: Retry number.
            retry_after: Server-provided ``Retry-After`` seconds (honoured if given).

        Returns:
            Seconds to sleep.
        """
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.max_delay * 2)
        upper = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return random.uniform(0, upper) + 0.1 * upper


def call_with_retries(
    fn: Callable[[], R],
    *,
    is_retryable: Callable[[BaseException], bool],
    retry_after: Callable[[BaseException], float | None] | None = None,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[R, int]:
    """Call ``fn`` and retry transient failures.

    Uses ``tenacity`` when installed (exponential jitter + ``Retry-After``);
    otherwise an equivalent built-in loop.

    Args:
        fn: Zero-argument callable to run.
        is_retryable: Predicate deciding whether an exception is transient (429/5xx/timeout).
        retry_after: Extracts a server ``Retry-After`` delay from an exception.
        policy: Backoff policy.
        sleep: Sleep function (injectable for tests).

    Returns:
        ``(result, retries_used)``.

    Raises:
        BaseException: The last exception when retries are exhausted or it is not retryable.
    """
    policy = policy or RetryPolicy()
    counter = {"n": 0}
    try:
        import tenacity
    except ImportError:
        tenacity = None  # type: ignore[assignment]

    if tenacity is not None:

        def _wait(state: Any) -> float:
            """Tenacity wait callback honouring Retry-After.

            Args:
                state: Tenacity retry state.

            Returns:
                Seconds to wait.
            """
            exc = state.outcome.exception() if state.outcome else None
            ra = retry_after(exc) if (retry_after and exc) else None
            return policy.delay(state.attempt_number, ra)

        def _before_sleep(state: Any) -> None:
            """Count and log retries.

            Args:
                state: Tenacity retry state.
            """
            counter["n"] += 1
            exc = state.outcome.exception() if state.outcome else None
            logger.info("Retry %d after %s", counter["n"], type(exc).__name__ if exc else "?")

        retrying = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(policy.max_retries + 1),
            wait=_wait,
            retry=tenacity.retry_if_exception(is_retryable),
            before_sleep=_before_sleep,
            sleep=sleep,
            reraise=True,
        )
        result = retrying(fn)
        return result, counter["n"]

    attempt = 0
    while True:
        try:
            return fn(), attempt
        except Exception as exc:
            if attempt >= policy.max_retries or not is_retryable(exc):
                raise
            attempt += 1
            ra = retry_after(exc) if retry_after else None
            delay = policy.delay(attempt, ra)
            logger.info("Retry %d in %.1fs after %s", attempt, delay, type(exc).__name__)
            sleep(delay)


def run_parallel(
    items: Iterable[T],
    worker: Callable[[T], R],
    *,
    max_workers: int,
    on_result: Callable[[int, T, R | None, BaseException | None], None],
) -> None:
    """Run ``worker`` over items in a thread pool; deliver results on the caller thread.

    ``on_result(position, item, result, error)`` is called from the calling
    (main) thread as futures complete, so UI updates stay single-threaded.
    Results carry their position so callers can store them deterministically.

    Args:
        items: Work items.
        worker: Function applied to each item (may raise).
        max_workers: Thread count (``1`` runs sequentially without a pool).
        on_result: Callback receiving each outcome.

    Raises:
        KeyboardInterrupt: Propagated after cancelling pending futures.
    """
    work = list(items)
    if max_workers <= 1:
        for pos, item in enumerate(work):
            try:
                res = worker(item)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate per-record failures
                on_result(pos, item, None, exc)
            else:
                on_result(pos, item, res, None)
        return
    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="translate")
    futures = {pool.submit(worker, item): pos for pos, item in enumerate(work)}
    try:
        for fut in as_completed(futures):
            pos = futures[fut]
            exc = fut.exception()
            on_result(pos, work[pos], None if exc else fut.result(), exc)
    except KeyboardInterrupt:
        for fut in futures:
            fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
