import os
from typing_extensions import override
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from sqa_system.core.language_model.enums.llm_enums import (
    ValidationResult,
    EndpointEnvVariable,
    EndpointType
)
from sqa_system.core.language_model.base.embedding_adapter import EmbeddingAdapter
from sqa_system.core.language_model.errors.api_key_missing_error import APIKeyMissingError


class GoogleAIEmbeddingAdapter(EmbeddingAdapter):
    """
    An implementation of the EmbeddingAdapter interface for Hugging Face Embeddings.
    """
    
    @override
    def prepare(self):
        """Prepares the embedding for use"""
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError
        self.embedding = GoogleGenerativeAIEmbeddings(
            model=self.embedding_config.name_model,
            request_options={"timeout": 10, "retries": 3})

    @override
    def validate(self) -> ValidationResult:
        """Validates if the embedding is ready to be used."""
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.GOOGLEAI).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
