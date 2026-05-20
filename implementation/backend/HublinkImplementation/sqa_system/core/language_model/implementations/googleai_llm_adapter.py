import os
from typing_extensions import override
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from sqa_system.core.language_model.enums.llm_enums import (
    ValidationResult,
    EndpointEnvVariable,
    EndpointType
)
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.language_model.errors.api_key_missing_error import APIKeyMissingError


class GoogleAiLLMAdapter(LLMAdapter):
    """Implementation of LLMAdapter for OpenAI LLMs."""

    @override
    def prepare(self):
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError

        max_tokens = self.llm_config.max_tokens
        if max_tokens == -1:
            max_tokens = None
        
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.14,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )
        
        self._set_llm(ChatGoogleGenerativeAI(
            model=self.llm_config.name_model,
            temperature=self.llm_config.temperature,
            max_tokens=max_tokens,
            timeout=140,
            rate_limiter=rate_limiter,
            max_retries=3,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        ))

    @override
    def validate(self) -> ValidationResult:
        """
        Validates if the LLM is ready to be used.
        """
        # check if environment variable is set
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.GOOGLEAI).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
