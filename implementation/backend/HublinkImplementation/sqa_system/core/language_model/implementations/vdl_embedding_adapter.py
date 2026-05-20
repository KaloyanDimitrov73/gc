import os
from typing_extensions import override
from langchain_ollama import OllamaEmbeddings

from sqa_system.core.language_model.enums.llm_enums import (
    ValidationResult,
    EndpointEnvVariable,
    EndpointType
)
from sqa_system.core.language_model.base.embedding_adapter import EmbeddingAdapter
from sqa_system.core.language_model.errors.api_key_missing_error import APIKeyMissingError


class VDLEmbeddingAdapter(EmbeddingAdapter):
    """
    Implementation of EmbeddingAdapter for KIT Virtual Design Lab Server.

    The VDL server exposes Ollama's embed API behind Open WebUI.
    See: https://sdq.kastel.kit.edu/wiki/Virtual_Design_Lab_Server
    """

    VDL_BASE_URL = "https://chat.vdl.sdq.kastel.kit.edu/ollama"

    @override
    def prepare(self):
        """Prepares the embedding for use"""
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError

        api_key = os.environ.get(EndpointEnvVariable.VDL_API_KEY.value)

        self.embedding = OllamaEmbeddings(
            base_url=self.VDL_BASE_URL,
            model=self.embedding_config.name_model,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {api_key}"}
            },
        )

    @override
    def validate(self) -> ValidationResult:
        """Validates if the embedding is ready to be used."""
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.VDL).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
