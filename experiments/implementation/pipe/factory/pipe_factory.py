import enum
from implementation.config.models.retrieval.kg_retrieval_config import KGRetrievalConfig
from implementation.config.models.retrieval.document_retrieval_config import DocumentRetrievalConfig
from implementation.config.models.pipe.generation_config import GenerationConfig
from implementation.config.models.pipe.pipe_config import PipeConfig
from implementation.config.models.pipe.post_retrieval_config import PostRetrievalConfig
from implementation.config.models.pipe.pre_retrieval_config import PreRetrievalConfig
from implementation.core.logging import get_logger

from ..base.pipe import Pipe
from ..generation.generation_pipe import GenerationPipe
from ..retrieval.implementations.kg_retrieval_pipe import KGRetrievalPipe
from ..retrieval.implementations.document_retrieval_pipe import DocumentRetrievalPipe
from ..post_retrieval.reranking_pipe import ReRankingPipe
from ..pre_retrieval.question_augmentation_pipe import QuestionAugmentationPipe

logger = get_logger(__name__)

class PostProcessingTechnique(enum.Enum):
    """An enumeration of the post-processing techniques."""
    RERANKING = "reranking"

class PreProcessingTechnique(enum.Enum):
    """An enumeration of the pre-processing techniques."""
    AUGMENTATION = "augmentation"


class PipeFactory:
    """A class that creates pipes based on the specified configuration."""

    @staticmethod
    def get_pipe(config: PipeConfig) -> Pipe:
        """
        Returns a Pipe instance based on the provided PipeConfig.

        Args:
            config (PipeConfig): The configuration for the pipe.
            
        Returns:
            Pipe: An instance of the appropriate Pipe subclass based on the configuration.
        """
        if isinstance(config, GenerationConfig):
            return GenerationPipe(config)
        if isinstance(config, KGRetrievalConfig):
            return KGRetrievalPipe(config)
        if isinstance(config, DocumentRetrievalConfig):
            return DocumentRetrievalPipe(config)
        if isinstance(config, PostRetrievalConfig):
            if config.post_technique == "reranking":
                return ReRankingPipe(config)
        if isinstance(config, PreRetrievalConfig):
            if config.pre_technique == "augmentation":
                return QuestionAugmentationPipe(config)

        raise ValueError("Pipe config has not valid type")
