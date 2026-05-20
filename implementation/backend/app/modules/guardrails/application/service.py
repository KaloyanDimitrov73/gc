"""
Guardrails service for validating user input and LLM output.
"""
from typing import Tuple, List
import logging

from .validators.base import Validator, is_infrastructure_error

logger = logging.getLogger(__name__)


class GuardrailsService:
    """
    Service for validating queries and LLM responses using guardrails-ai.

    Validators are injected at construction time, making the service fully
    unit-testable without a real guardrails-ai installation.

    Input validators: Block harmful or policy-violating queries before calling HubLink.
    Output validators: Flag issues in LLM responses after the call (never block).
    """

    def __init__(
        self,
        input_validators: List[Validator],
        output_validators: List[Validator],
    ) -> None:
        self._input_validators = input_validators
        self._output_validators = output_validators

    def is_available(self) -> bool:
        """True if at least one validator is operational."""
        all_validators = self._input_validators + self._output_validators
        return any(v.is_available() for v in all_validators)

    async def validate_input(self, question: str) -> Tuple[bool, str]:
        """
        Validate user input before sending to HubLink.

        Returns:
            Tuple of (is_valid, rejection_reason).
            If is_valid is False, rejection_reason explains why.
        """
        for v in self._input_validators:
            if not v.is_available():
                continue
            try:
                passed, reason = await v.validate(question)
                if not passed:
                    logger.info("Input rejected by %s: %s", v.name, reason)
                    return False, reason
            except Exception as e:
                if is_infrastructure_error(str(e)):
                    logger.warning(
                        "Guardrails input validation skipped due to infrastructure/configuration issue (%s): %s",
                        v.name, e,
                    )
                    continue
                raise
        return True, ""

    async def validate_output(self, answer: str) -> Tuple[str, bool, str]:
        """
        Validate LLM output after receiving from HubLink.

        Returns:
            Tuple of (answer, validation_passed, warning_message).
            The answer is always returned unchanged.
            warning_message is empty if validation passed.
        """
        for v in self._output_validators:
            if not v.is_available():
                continue
            try:
                passed, _ = await v.validate(answer)
                if not passed:
                    warning = "The response may contain issues flagged by quality checks."
                    logger.warning("Output flagged by %s", v.name)
                    return answer, False, warning
            except Exception as e:
                if is_infrastructure_error(str(e)):
                    logger.warning(
                        "Guardrails output validation skipped due to infrastructure/configuration issue (%s): %s",
                        v.name, e,
                    )
                    continue
                logger.error("Output validation error in %s: %s", v.name, e, exc_info=True)
        return answer, True, ""
