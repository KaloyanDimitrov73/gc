import os
from typing_extensions import override
from langchain_openai import ChatOpenAI

from language_model.enums.llm_enums import (
    ValidationResult,
    EndpointEnvVariable,
    EndpointType
)
from language_model.base.llm_adapter import LLMAdapter
from language_model.errors.api_key_missing_error import APIKeyMissingError


class OpenAiLLMAdapter(LLMAdapter):
    """Implementation of LLMAdapter for OpenAI LLMs."""

    @override
    def prepare(self):
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError

        max_tokens = self.llm_config.max_tokens
        if max_tokens == -1:
            max_tokens = None
        self._set_llm(ChatOpenAI(
            model=self.llm_config.name_model,
            temperature=self.llm_config.temperature,
            max_tokens=max_tokens,
            timeout=140,
            reasoning_effort=self.llm_config.reasoning_effort
        ))

    @override
    def validate(self) -> ValidationResult:
        """
        Validates if the LLM is ready to be used.
        """
        # check if environment variable is set
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.OPENAI).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
