"""
Validator that detects gibberish/incoherent text.

NOTE: This validator is intentionally excluded from the default service configuration
in dependencies.py due to too many false positives on legitimate domain-specific queries.
It is available for opt-in use and experimentation.
"""
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class GibberishValidator:
    name = "GibberishText"

    def __init__(self) -> None:
        self._available = False
        self._guard = None
        try:
            from guardrails import Guard
            from guardrails.hub import GibberishText

            self._guard = Guard().use(GibberishText(on_fail="noop"))
            self._available = True
            logger.info("GibberishValidator initialized successfully")
        except Exception as e:
            logger.warning("GibberishValidator unavailable: %s", e)

    def is_available(self) -> bool:
        return self._available

    def validate(self, text: str) -> Tuple[bool, str]:
        result = self._guard.validate(text)
        if result.validation_passed:
            return True, ""
        return False, f"Validation failed for {self.name}"
