import os
from typing_extensions import override
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from sqa_system.core.language_model.enums.llm_enums import (
    ValidationResult,
    EndpointEnvVariable,
    EndpointType
)
from sqa_system.core.language_model.base.embedding_adapter import EmbeddingAdapter
from sqa_system.core.language_model.errors.api_key_missing_error import APIKeyMissingError


class HuggingFaceEmbeddingAdapter(EmbeddingAdapter):
    """
    An implementation of the EmbeddingAdapter interface for Hugging Face Embeddings.
    """
    _instances = {}
    _initialized = False

    def __new__(cls, embedding_config):
        config_key = embedding_config.config_hash
        if config_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[config_key] = instance
        return cls._instances[config_key]

    def __init__(self, embedding_config):
        super().__init__(embedding_config)
        if not self._initialized:
            self._model_is_loaded = False
            self.prepare()
            self._initialized = True

    @override
    def prepare(self):
        """Prepares the embedding for use"""
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError
        if not self._model_is_loaded:
            self.embedding = HuggingFaceEmbeddings(
                model_name=self.embedding_config.name_model,
                # we need to run this on the cpu as using the embedding
                # with the LLM on the GPU will cause an OutOfMemoryError
                model_kwargs={"device": "cpu"}, 
            )
            self._model_is_loaded = True

    @override
    def validate(self) -> ValidationResult:
        """Validates if the embedding is ready to be used."""
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.HUGGINGFACE).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        return ValidationResult.VALID
