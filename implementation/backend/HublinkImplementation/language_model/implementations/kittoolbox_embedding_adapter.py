import os
from typing_extensions import override
from langchain_openai import OpenAIEmbeddings

from language_model.enums.llm_enums import (
    ValidationResult,
    EndpointEnvVariable,
    EndpointType
)
from language_model.base.embedding_adapter import EmbeddingAdapter
from language_model.errors.api_key_missing_error import APIKeyMissingError


class KitToolboxEmbeddingAdapter(EmbeddingAdapter):
    """
    Implementation of EmbeddingAdapter for the KIT KI-Toolbox.

    The KI-Toolbox exposes an OpenAI-compatible embeddings endpoint (Open WebUI).
    See: https://ki-toolbox.scc.kit.edu/
    """

    KIT_TOOLBOX_BASE_URL = "https://ki-toolbox.scc.kit.edu/api"

    @override
    def prepare(self):
        """Prepares the embedding for use"""
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError

        api_key = os.environ.get(EndpointEnvVariable.KIT_TOOLBOX_API_KEY.value)

        self.embedding = OpenAIEmbeddings(
            base_url=self.KIT_TOOLBOX_BASE_URL,
            api_key=api_key,
            model=self.embedding_config.name_model,
        )

    @override
    def validate(self) -> ValidationResult:
        """Validates if the embedding is ready to be used."""
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.KIT_TOOLBOX).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
