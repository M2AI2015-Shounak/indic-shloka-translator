"""Install missing Python packages on demand (used when a model is selected in the widget)."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# import name → pip specifiers tried in order
PACKAGE_SOURCES: dict[str, list[list[str]]] = {
    "ollama": [["ollama>=0.4,<1"]],
    "IndicTransToolkit": [
        ["IndicTransToolkit>=1.0"],
        ["cython", "IndicTransToolkit>=1.0"],
        ["git+https://github.com/VarunGumma/IndicTransToolkit.git"],
    ],
    "transformers": [["transformers>=4.51,<5"]],
    "sentencepiece": [["sentencepiece>=0.2,<0.3"]],
}


def is_importable(module: str) -> bool:
    """Whether a module can be imported (without importing it).

    Args:
        module: Top-level import name, e.g. ``"ollama"``.

    Returns:
        ``True`` if importable.

    Example:
        >>> is_importable("json")
        True
    """
    importlib.invalidate_caches()
    return importlib.util.find_spec(module) is not None


def ensure_packages(modules: list[str]) -> dict[str, str]:
    """Pip-install any missing modules, trying each known source in turn.

    Args:
        modules: Import names to make available.

    Returns:
        Mapping of module → error text for modules that are still missing
        (empty when everything is importable).
    """
    failures: dict[str, str] = {}
    for module in modules:
        if is_importable(module):
            continue
        sources = PACKAGE_SOURCES.get(module, [[module]])
        last_error = ""
        for spec in sources:
            print(f"Installing {' '.join(spec)} …")
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *spec],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and is_importable(module):
                logger.info("Installed %s via %s", module, " ".join(spec))
                break
            lines = [ln for ln in (proc.stderr or proc.stdout).splitlines() if ln.strip()]
            last_error = " | ".join(lines[-3:])
        else:
            failures[module] = last_error or "installation failed"
            logger.warning("Could not install %s: %s", module, failures[module])
    return failures
