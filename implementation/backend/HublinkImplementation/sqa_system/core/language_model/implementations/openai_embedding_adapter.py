import os

from typing_extensions import override
from langchain_openai import OpenAIEmbeddings

from sqa_system.core.language_model.errors.api_key_missing_error import APIKeyMissingError
from sqa_system.core.language_model.enums.llm_enums import (
    ValidationResult,
    EndpointEnvVariable,
    EndpointType
)
from sqa_system.core.language_model.base.embedding_adapter import EmbeddingAdapter


class OpenAiEmbeddingAdapter(EmbeddingAdapter):
    """
    An implementation of the EmbeddingAdapter interface for OpenAI Embeddings.
    """

    @override
    def prepare(self):
        """Prepares the embedding for use"""
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError
        self.embedding = OpenAIEmbeddings(
            model=self.embedding_config.name_model)

    @override
    def validate(self) -> ValidationResult:
        """Validates if the embedding is ready to be used."""
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.OPENAI).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
