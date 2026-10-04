"""Secret retrieval (Colab Secrets → environment → getpass) and log redaction.

Secrets are never written to config, logs, notebooks or output files. Every
handler on the root logger gets a :class:`RedactingFilter` that masks OpenAI
(``sk-...``) and Hugging Face (``hf_...``) tokens as well as any secret value
that was loaded through :func:`get_secret`.
"""

from __future__ import annotations

import getpass
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
)
REDACTED = "[REDACTED]"

_known_secret_values: set[str] = set()
_lock = threading.Lock()


def redact(text: str) -> str:
    """Mask every secret-looking substring in ``text``.

    Args:
        text: Arbitrary text (log message, exception string, ...).

    Returns:
        The text with secrets replaced by ``[REDACTED]``.

    Example:
        >>> redact("key=sk-" + "a" * 16)
        'key=[REDACTED]'
    """
    if not text:
        return text
    out = text
    for pattern in SECRET_PATTERNS:
        out = pattern.sub(REDACTED, out)
    with _lock:
        values = list(_known_secret_values)
    for value in values:
        if value and len(value) >= 6:
            out = out.replace(value, REDACTED)
    return out


class RedactingFilter(logging.Filter):
    """Logging filter that redacts secrets from the formatted message."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite ``record.msg`` with secrets masked; never drops the record.

        Args:
            record: The log record being emitted.

        Returns:
            Always ``True`` (the record is kept).
        """
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = str(record.msg)
        record.msg = redact(message)
        record.args = ()
        return True


def install_redaction(target: logging.Logger | None = None) -> None:
    """Attach :class:`RedactingFilter` to a logger and all its handlers.

    Args:
        target: Logger to protect; defaults to the root logger.
    """
    log = target or logging.getLogger()
    if not any(isinstance(f, RedactingFilter) for f in log.filters):
        log.addFilter(RedactingFilter())
    for handler in log.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())


def register_secret(value: str | None) -> None:
    """Remember a secret value so it is masked even if it lacks a known prefix.

    Args:
        value: Secret string; ``None``/empty is ignored.
    """
    if value:
        with _lock:
            _known_secret_values.add(value)


def _from_colab(name: str) -> str | None:
    """Read a secret from Colab's ``userdata`` store if running in Colab.

    Args:
        name: Secret name, e.g. ``"OPENAI_API_KEY"``.

    Returns:
        The secret value or ``None`` when unavailable / access not granted.
    """
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        value = userdata.get(name)
    except Exception as exc:  # noqa: BLE001 - Colab raises several custom errors
        logger.debug("Colab secret %s not available: %s", name, type(exc).__name__)
        return None
    return value or None


def get_secret(name: str, *, interactive: bool = False, required: bool = False) -> str | None:
    """Retrieve a secret: Colab Secrets → environment variable → ``getpass``.

    The value is registered for log redaction and never logged.

    Args:
        name: Secret name such as ``"OPENAI_API_KEY"`` or ``"HF_TOKEN"``.
        interactive: If ``True`` and no other source has it, prompt with ``getpass``.
        required: If ``True``, raise when the secret cannot be found.

    Returns:
        The secret string, or ``None`` when not found and not required.

    Raises:
        RuntimeError: If ``required`` is ``True`` and the secret is missing.

    Example:
        >>> key = get_secret("OPENAI_API_KEY")  # doctest: +SKIP
    """
    colab_value = _from_colab(name)
    value = colab_value or os.environ.get(name) or None
    source = "colab" if colab_value else ("env" if value else None)
    if not value and interactive:
        try:
            value = getpass.getpass(f"Enter {name} (input hidden, leave blank to skip): ").strip() or None
            source = "getpass" if value else None
        except (EOFError, OSError):
            value = None
    if value:
        register_secret(value)
        os.environ.setdefault(name, value)
        logger.info("Secret %s loaded (source: %s)", name, source)
    elif required:
        raise RuntimeError(f"Required secret {name} not found in Colab Secrets, environment or prompt")
    else:
        logger.info("Secret %s not set", name)
    return value


def load_dotenv_if_present(path: str | os.PathLike[str] | None = None) -> bool:
    """Load a local ``.env`` file (development convenience; never committed).

    Args:
        path: Path to the ``.env`` file; default lets python-dotenv search.

    Returns:
        ``True`` if a file was loaded, else ``False``.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(dotenv_path=path, override=False))
