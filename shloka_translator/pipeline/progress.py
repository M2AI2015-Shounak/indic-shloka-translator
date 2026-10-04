"""Percent-completion progress display (ipywidgets in Colab, tqdm elsewhere).

Label format::

    63.4% complete | 136/215 | ✓ 130 Success | ⤼ 4 Skipped | ✗ 2 Failed | 3.1 rec/s | ETA 00:00:25

All updates happen on the main thread (the executor delivers results there).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from shloka_translator.utils.timing import format_hms

logger = logging.getLogger(__name__)


def in_notebook() -> bool:
    """Detect whether we run inside a Jupyter/Colab kernel.

    Returns:
        ``True`` inside a notebook kernel.
    """
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


@dataclass
class ProgressState:
    """Counters behind the progress label.

    Attributes:
        total: Records to account for.
        done: Records finished (any status).
        success: Success count.
        skipped: Skipped count.
        failed: Failed count.
        started: Monotonic start time.
        processed_live: Records actually processed in this run (rate/ETA basis).
    """

    total: int
    done: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    started: float = field(default_factory=time.monotonic)
    processed_live: int = 0

    def label(self) -> str:
        """Render the status line.

        Returns:
            Formatted label.

        Example:
            >>> ProgressState(total=4, done=2, success=2).label().startswith("50.0% complete | 2/4")
            True
        """
        pct = 100.0 * self.done / self.total if self.total else 100.0
        elapsed = max(1e-6, time.monotonic() - self.started)
        rate = self.processed_live / elapsed if self.processed_live else 0.0
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 else (0 if remaining == 0 else None)
        return (
            f"{pct:.1f}% complete | {self.done}/{self.total} | ✓ {self.success} Success | "
            f"⤼ {self.skipped} Skipped | ✗ {self.failed} Failed | {rate:.1f} rec/s | ETA {format_hms(eta)}"
        )


class ProgressReporter:
    """Progress bar for one model's records."""

    def __init__(self, total: int, title: str = "", *, use_widgets: bool | None = None) -> None:
        """Create the display.

        Args:
            total: Number of records.
            title: Prefix (e.g. model name).
            use_widgets: Force widgets on/off (default: auto-detect notebook).
        """
        self.state = ProgressState(total=total)
        self.title = title
        self._bar = None
        self._html = None
        self._tqdm = None
        widgets_wanted = in_notebook() if use_widgets is None else use_widgets
        if widgets_wanted:
            try:
                import ipywidgets as widgets
                from IPython.display import display

                self._bar = widgets.IntProgress(
                    value=0, min=0, max=max(1, total), bar_style="info", layout=widgets.Layout(width="100%")
                )
                self._html = widgets.HTML(value=self._render())
                display(widgets.VBox([widgets.HTML(f"<b>{title}</b>"), self._bar, self._html]))
            except ImportError:
                self._bar = None
        if self._bar is None:
            try:
                from tqdm.auto import tqdm

                self._tqdm = tqdm(total=total, desc=title or "records", unit="rec", leave=True)
            except ImportError:
                self._tqdm = None

    def _render(self) -> str:
        """HTML label.

        Returns:
            Label text.
        """
        return f"<code>{self.state.label()}</code>"

    def update(self, status: str, *, live: bool = True, n: int = 1) -> None:
        """Record finished records and refresh the display.

        Args:
            status: ``"Success"``, ``"Skipped"`` or ``"Failed"``.
            live: ``False`` for cached/skipped records (excluded from the rate).
            n: Number of records with this status.
        """
        s = self.state
        s.done += n
        if status == "Success":
            s.success += n
        elif status == "Skipped":
            s.skipped += n
        else:
            s.failed += n
        if live:
            s.processed_live += n
        if self._bar is not None:
            self._bar.value = s.done
            self._html.value = self._render()  # type: ignore[union-attr]
        elif self._tqdm is not None:
            self._tqdm.update(n)
            self._tqdm.set_postfix_str(s.label().split(" | ", 2)[2])
        elif s.done == s.total or s.done % max(1, s.total // 20) == 0:
            logger.info("%s %s", self.title, s.label())

    def close(self) -> None:
        """Finalise the display."""
        if self._bar is not None:
            self._bar.bar_style = "success" if self.state.failed == 0 else "warning"
            self._html.value = self._render()  # type: ignore[union-attr]
        if self._tqdm is not None:
            self._tqdm.close()
        logger.info("%s %s", self.title, self.state.label())


class MultiModelProgress:
    """Outer bar (models done / total) for ALL MODELS mode."""

    def __init__(self, models: list[str], *, use_widgets: bool | None = None) -> None:
        """Create the outer display.

        Args:
            models: Models to run.
            use_widgets: Force widgets on/off.
        """
        self.models = models
        self.done = 0
        self._outer = ProgressReporter(len(models), "Models", use_widgets=use_widgets)

    def model_done(self, ok: bool) -> None:
        """Advance the outer bar.

        Args:
            ok: Whether the model run completed.
        """
        self.done += 1
        self._outer.update("Success" if ok else "Failed")

    def close(self) -> None:
        """Close the outer bar."""
        self._outer.close()
