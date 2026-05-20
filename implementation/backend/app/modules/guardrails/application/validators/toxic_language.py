"""
Validator that detects toxic language in text.
"""
from typing import Tuple
import logging
import asyncio

from .base import ensure_nltk_resources

logger = logging.getLogger(__name__)


class ToxicLanguageValidator:
    name = "ToxicLanguage"

    def __init__(self) -> None:
        self._available = False
        self._guard = None
        try:
            ensure_nltk_resources()
            from guardrails import Guard
            from guardrails.hub import ToxicLanguage

            self._guard = Guard().use(ToxicLanguage(on_fail="noop"))
            self._available = True
            logger.info("ToxicLanguageValidator initialized successfully")
        except Exception as e:
            logger.warning("ToxicLanguageValidator unavailable: %s", e)

    def is_available(self) -> bool:
        return self._available

    async def validate(self, text: str) -> Tuple[bool, str]:
        result = await asyncio.to_thread(self._guard.validate, text)
        if result.validation_passed:
            return True, ""
        return False, f"Validation failed for {self.name}"
