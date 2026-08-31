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


class KitToolboxLLMAdapter(LLMAdapter):
    """
    Implementation of LLMAdapter for the KIT KI-Toolbox.

    The KI-Toolbox provides an OpenAI-compatible API (Open WebUI) fronting
    models from several providers (KIT-local, Azure, Google, MistralAI).
    See: https://ki-toolbox.scc.kit.edu/
    """

    KIT_TOOLBOX_BASE_URL = "https://ki-toolbox.scc.kit.edu/api"

    @override
    def prepare(self):
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError

        max_tokens = self.llm_config.max_tokens
        if max_tokens == -1:
            max_tokens = None

        api_key = os.environ.get(EndpointEnvVariable.KIT_TOOLBOX_API_KEY.value)

        self._set_llm(ChatOpenAI(
            base_url=self.KIT_TOOLBOX_BASE_URL,
            api_key=api_key,
            model=self.llm_config.name_model,
            temperature=self.llm_config.temperature,
            max_tokens=max_tokens,
            timeout=140,
        ))

    @override
    def validate(self) -> ValidationResult:
        """
        Validates if the LLM is ready to be used.
        """
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.KIT_TOOLBOX).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
