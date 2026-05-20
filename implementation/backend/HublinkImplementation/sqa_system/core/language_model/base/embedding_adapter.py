from abc import ABC, abstractmethod
import time
from typing import Optional
from langchain_core.embeddings.embeddings import Embeddings
from sqa_system.core.language_model.enums.llm_enums import ValidationResult
from sqa_system.core.config.models.embedding_config import EmbeddingConfig
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


class EmbeddingAdapterNotReadyError(Exception):
    """Error raised when embedding adapter is not ready."""


class EmbeddingAdapter(ABC):
    """
    An interface for a Embedding Model.
    Adapts the Embedding Model interface from LangChain.
    
    Args:
        embedding_config: Configuration for the embedding model.
    """

    def __init__(self, embedding_config: EmbeddingConfig):
        self.embedding_config = embedding_config
        self.embedding: Optional[Embeddings] = None

    def get_embeddings(self) -> Embeddings:
        """Returns the LangChain Embeddings object."""
        if self.embedding is None:
            raise EmbeddingAdapterNotReadyError("Embedding Adapter not ready")
        return self.embedding

    def embed(self, text: str) -> list[float] | None:
        """
        Embeds the given text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding or
            None if the embedding fails.
        """
        retry_amount = 3
        for i in range(retry_amount):
            try:
                start = time.time()
                result = self.get_embeddings().embed_query(text)
                elapsed = time.time() - start
                logger.info(f"Embedding completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                logger.warning(f"Error during embedding: {e} Trying again...")
                if i == retry_amount - 1:
                    raise e
                time.sleep(80)
        return None

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """
        Embeds the given list of texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of lists of floats representing the embeddings or
            None if the embedding fails.
        """
        return self.get_embeddings().embed_documents(texts)

    @abstractmethod
    def validate(self) -> ValidationResult:
        """
        A abstract method that needs to be implemented by the child class.
        It should validate whether the embedding adapter is ready to use.
        
        Returns:
            ValidationResult: Result of the validation.
        """

    @abstractmethod
    def prepare(self):
        """
        A abstract method that needs to be implemented by the child class.
        It should prepare the embedding adapter for use.
        """
