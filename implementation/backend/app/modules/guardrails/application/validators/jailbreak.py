"""
Validator that detects jailbreak attempts in text.
"""
from typing import Tuple
import logging
import asyncio

from .base import ensure_nltk_resources

logger = logging.getLogger(__name__)


class JailbreakValidator:
    name = "DetectJailbreak"

    def __init__(self) -> None:
        self._available = False
        self._guard = None
        try:
            ensure_nltk_resources()
            from guardrails import Guard
            from guardrails.hub import DetectJailbreak

            self._guard = Guard().use(DetectJailbreak(on_fail="noop"))
            self._available = True
            logger.info("JailbreakValidator initialized successfully")
        except Exception as e:
            logger.warning("JailbreakValidator unavailable: %s", e)

    def is_available(self) -> bool:
        return self._available

    async def validate(self, text: str) -> Tuple[bool, str]:
        result = await asyncio.to_thread(self._guard.validate, text)
        if result.validation_passed:
            return True, ""
        return False, f"Validation failed for {self.name}"
