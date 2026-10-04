"""Time-zone aware timestamps and simple timing helpers."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def now_in_tz(timezone: str = "Asia/Kolkata") -> datetime:
    """Return the current time in the configured time zone.

    Args:
        timezone: IANA zone name such as ``"Asia/Kolkata"``.

    Returns:
        Aware :class:`datetime`.
    """
    return datetime.now(ZoneInfo(timezone))


def format_hms(seconds: float | None) -> str:
    """Format a duration as ``HH:MM:SS``.

    Args:
        seconds: Duration in seconds; ``None`` or negative gives ``"--:--:--"``.

    Returns:
        Formatted string.

    Example:
        >>> format_hms(3725)
        '01:02:05'
    """
    if seconds is None or seconds < 0 or math.isnan(seconds):
        return "--:--:--"
    seconds = round(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def percentile(values: list[float], pct: float) -> float | None:
    """Compute a percentile with linear interpolation.

    Args:
        values: Sample values.
        pct: Percentile in ``[0, 100]``.

    Returns:
        The percentile, or ``None`` when ``values`` is empty.

    Example:
        >>> percentile([1, 2, 3, 4], 50)
        2.5
    """
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class Stopwatch:
    """Monotonic stopwatch usable as a context manager.

    Example:
        >>> with Stopwatch() as sw:
        ...     pass
        >>> sw.elapsed >= 0
        True
    """

    def __init__(self) -> None:
        """Create a stopped stopwatch."""
        self._start: float | None = None
        self._end: float | None = None

    def __enter__(self) -> Stopwatch:
        """Start timing.

        Returns:
            This stopwatch.
        """
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop timing.

        Args:
            *exc: Exception info (ignored).
        """
        self.stop()

    def start(self) -> None:
        """Start (or restart) the stopwatch."""
        self._start = time.perf_counter()
        self._end = None

    def stop(self) -> float:
        """Stop the stopwatch.

        Returns:
            Elapsed seconds.
        """
        self._end = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed seconds so far (or until :meth:`stop`).

        Returns:
            Seconds as float; ``0.0`` if never started.
        """
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start
