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


class VDLLLMAdapter(LLMAdapter):
    """
    Implementation of LLMAdapter for KIT Virtual Design Lab Server.

    The VDL server provides an OpenAI-compatible API running on Ollama + Open WebUI.
    See: https://sdq.kastel.kit.edu/wiki/Virtual_Design_Lab_Server
    """

    VDL_BASE_URL = "https://chat.vdl.sdq.kastel.kit.edu/api"

    @override
    def prepare(self):
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError

        max_tokens = self.llm_config.max_tokens
        if max_tokens == -1:
            max_tokens = None

        api_key = os.environ.get(EndpointEnvVariable.VDL_API_KEY.value)

        self._set_llm(ChatOpenAI(
            base_url=self.VDL_BASE_URL,
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
            EndpointType.VDL).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
