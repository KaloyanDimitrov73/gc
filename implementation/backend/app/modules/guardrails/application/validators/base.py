"""
Base protocol and shared utilities for guardrails validators.
"""
from typing import Protocol, Tuple
import logging

logger = logging.getLogger(__name__)


class Validator(Protocol):
    """Unified interface for input and output validators."""

    name: str

    def is_available(self) -> bool: ...

    async def validate(self, text: str) -> Tuple[bool, str]:
        """
        Validate the given text.

        Returns:
            Tuple of (passed, reason).
            If passed is False, reason explains why.
        Raises:
            Exception: On infrastructure/configuration errors (not user-input failures).
        """
        ...


def is_infrastructure_error(reason: str) -> bool:
    """Return True when a validator failure is infra/configuration, not user content."""
    normalized = reason.lower()
    markers = (
        "remote inference unauthorized",
        "please run `guardrails configure`",
        "resource punkt_tab not found",
        "resource punkt not found",
        "lookuperror",
        "nltk",
    )
    return any(m in normalized for m in markers)


def ensure_nltk_resources() -> None:
    """Download NLTK tokenizer resources required by some validators."""
    import nltk

    def _ensure(resource_path: str, download_name: str) -> None:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", download_name)
            nltk.download(download_name, quiet=True)

    _ensure("tokenizers/punkt", "punkt")
    _ensure("tokenizers/punkt_tab/english", "punkt_tab")
