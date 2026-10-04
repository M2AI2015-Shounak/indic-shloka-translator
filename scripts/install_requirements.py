"""Fault-tolerant installer for requirements files (used by the Colab notebooks).

``pip install -r requirements.txt`` is all-or-nothing: if a single package
fails to build (common on new Python versions for packages with C extensions,
e.g. ``fasttext-wheel`` or ``IndicTransToolkit``), pip installs *nothing* and
every backend then reports "... package not installed". This script:

1. tries the whole file in one pip call (fast path);
2. if that fails, installs each requirement on its own, so one failure no
   longer blocks the others;
3. prints a ✓/✗ table with the tail of pip's error for every failure;
4. exits 0 when every *core* package is installed (optional failures only
   disable the feature that needs them).

Usage::

    python scripts/install_requirements.py requirements.txt
    python scripts/install_requirements.py requirements-train.txt
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Packages whose failure only disables one optional feature (not the whole app).
OPTIONAL = {
    "fasttext-wheel": "fastText language-ID fallback and IndicLID (rule-based detection still works)",
    "sanskrit_parser": "rule-based Sanskrit sandhi splitting (LLM helpers still segment)",
    "aksharamukha": "extra transliteration route (indic-transliteration is used first)",
    "indic-nlp-library": "rule-based tokenisation (whitespace fallback is used)",
    "IndicTransToolkit": "IndicTrans2 and the custom model",
    "ollama": "Ollama models",
    "nvidia-ml-py": "GPU metrics sheet",
    "bitsandbytes": "8-bit optimisers (training still works)",
    "pytest": "running the test-suite",
}

# Alternative sources tried, in order, when the PyPI requirement fails to install.
FALLBACKS: dict[str, list[list[str]]] = {
    # builds a Cython extension; make sure Cython is present, then try the GitHub source
    "IndicTransToolkit": [
        ["cython", "IndicTransToolkit"],
        ["git+https://github.com/VarunGumma/IndicTransToolkit.git"],
    ],
    # wheels may be missing for the newest Python; NumPy-2 builds exist under other names
    "fasttext-wheel": [["fasttext-numpy2-wheel"], ["fasttext"]],
    "sanskrit_parser": [["sanskrit_parser", "--no-deps"]],
}


def read_requirements(path: Path, seen: set[Path] | None = None) -> list[str]:
    """Read requirement lines, following ``-r other.txt`` includes.

    Args:
        path: Requirements file.
        seen: Files already read (prevents include loops).

    Returns:
        Requirement specifiers without comments.
    """
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    reqs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            reqs += read_requirements(path.parent / line.split(maxsplit=1)[1], seen)
        else:
            reqs.append(line)
    return reqs


def package_name(spec: str) -> str:
    """Extract the distribution name from a requirement specifier.

    Args:
        spec: e.g. ``"transformers>=4.51,<5"``.

    Returns:
        e.g. ``"transformers"``.
    """
    return re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0]


def pip(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m pip install`` quietly.

    Args:
        args: Requirement specifiers or ``-r file``.

    Returns:
        Completed process (stdout/stderr captured).
    """
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def main(argv: list[str]) -> int:
    """Install a requirements file robustly and report per-package results.

    Args:
        argv: Command-line arguments (first: requirements file).

    Returns:
        0 if all core packages installed, 1 otherwise.
    """
    req_file = Path(argv[0] if argv else "requirements.txt")
    specs = read_requirements(req_file)
    print(f"Installing {len(specs)} requirements from {req_file} …")
    whole = pip(["-r", str(req_file)])
    if whole.returncode == 0:
        print("✓ all requirements installed in one pass")
        return 0
    print("⚠ The one-pass install failed (pip installs nothing when one package fails).")
    print("  Installing packages one by one so a single failure cannot block the rest…\n")
    failures: dict[str, str] = {}
    for spec in specs:
        name = package_name(spec)
        res = pip([spec])
        via = ""
        if res.returncode != 0:
            for alt in FALLBACKS.get(name, []):
                alt_res = pip(alt)
                if alt_res.returncode == 0:
                    res, via = alt_res, f"  (via fallback: {' '.join(alt)})"
                    break
        if res.returncode == 0:
            print(f"  ✓ {spec}{via}")
        else:
            tail = [ln for ln in (res.stderr or res.stdout).strip().splitlines() if ln.strip()][-4:]
            failures[name] = "\n      ".join(tail)
            print(f"  ✗ {spec}\n      {failures[name]}")
    if not failures:
        print("\n✓ all requirements installed")
        return 0
    core_failed = [n for n in failures if n not in OPTIONAL]
    print("\nSummary of failures:")
    for name in failures:
        impact = OPTIONAL.get(name, "CORE package — the app needs it")
        print(f"  ✗ {name}: affects {impact}")
    return 1 if core_failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
