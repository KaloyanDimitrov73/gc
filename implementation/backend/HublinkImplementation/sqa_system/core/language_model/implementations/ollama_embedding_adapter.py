from typing_extensions import override
from langchain_ollama import OllamaEmbeddings
from sqa_system.core.language_model.enums.llm_enums import ValidationResult

from sqa_system.core.language_model.base.embedding_adapter import EmbeddingAdapter


class OllamaEmbeddingAdapter(EmbeddingAdapter):
    """Implementation of the EmbeddingAdapter for Ollama Embedding Models."""

    @override
    def prepare(self):
        self.embedding = OllamaEmbeddings(
            model=self.embedding_config.name_model
        )

    @override
    def validate(self) -> ValidationResult:
        """
        Validates if the Embedding Model is ready to be used.
        """
        return ValidationResult.VALID
