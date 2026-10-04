"""Repository hygiene: no committed secrets, complete docstrings everywhere."""

from __future__ import annotations

import ast
import re

from tests.helpers import REPO

SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{10,}|hf_[A-Za-z0-9]{10,}")
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "Output_Folder", "logs", "models"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".ipynb", ".sh", ".json", ".jsonl", ".csv", ".example", ""}


def test_no_secret_strings_in_repo() -> None:
    """No file in the repo contains an OpenAI/HF token pattern."""
    offenders = []
    for path in REPO.rglob("*"):
        if not path.is_file() or any(p in SKIP_DIRS for p in path.relative_to(REPO).parts):
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in SECRET_RE.finditer(text):
            # allow the regex definitions themselves and test placeholders built at runtime
            if "{10,}" in text[max(0, m.start() - 5) : m.end() + 10]:
                continue
            offenders.append(f"{path.relative_to(REPO)}: {m.group(0)[:12]}…")
    assert not offenders, offenders


def _needs_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether a function has parameters other than self/cls.

    Args:
        node: Function node.

    Returns:
        ``True`` if an ``Args:`` section is required.
    """
    names = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    names = [n for n in names if n not in ("self", "cls")]
    return bool(names or node.args.vararg or node.args.kwarg)


def _needs_returns(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether a function needs a ``Returns:`` section (annotated non-None return).

    Args:
        node: Function node.

    Returns:
        ``True`` if required.
    """
    if node.name == "__init__" or node.returns is None:
        return False
    ann = ast.unparse(node.returns)
    return ann not in ("None", "NoReturn")


def test_every_function_has_complete_docstring() -> None:
    """Every module, class and function in the package has a Google-style docstring."""
    problems = []
    for path in [*sorted((REPO / "shloka_translator").rglob("*.py")), REPO / "scripts" / "run_cli.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO)
        if not ast.get_docstring(tree):
            problems.append(f"{rel}: module docstring missing")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
                problems.append(f"{rel}:{node.lineno} class {node.name}: no docstring")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if not doc:
                    problems.append(f"{rel}:{node.lineno} {node.name}: no docstring")
                    continue
                if _needs_args(node) and "Args:" not in doc:
                    problems.append(f"{rel}:{node.lineno} {node.name}: missing Args")
                if _needs_returns(node) and "Returns:" not in doc and "Yields:" not in doc:
                    problems.append(f"{rel}:{node.lineno} {node.name}: missing Returns")
    assert not problems, "\n".join(problems)
