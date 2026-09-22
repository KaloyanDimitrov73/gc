
from enum import Enum

from knowledge_base.config.chunking_strategy_config import ChunkingStrategyConfig
from knowledge_base.vector_store.chunking.base.chunking_strategy import ChunkingStrategy
from core.logging.logging import get_logger

from .strategies.recursiv_character_chunking_strategy import RecursiveCharacterChunkingStrategy
logger = get_logger(__name__)


class ChunkingStrategyType(Enum):
    """
    An enum class that defines the available chunking strategies.
    """
    RECURSIV_CHARACTER = "RecursiveCharacterChunkingStrategy"


class ChunkingStrategyFactory:
    """
    A factory class that creates chunking strategies based on the provided configuration.
    This class is responsible for instantiating the appropriate chunking strategy
    based on the type specified in the configuration.
    """

    @staticmethod
    def create(config: ChunkingStrategyConfig) -> ChunkingStrategy:
        """
        Creates a chunking strategy based on the specified configuration.

        Args:
            config (ChunkingStrategyConfig): The configuration for the chunking strategy.

        Returns:
            ChunkingStrategy: The created chunking strategy.
        """
        if config.chunking_strategy_type == ChunkingStrategyType.RECURSIV_CHARACTER.value:
            return RecursiveCharacterChunkingStrategy(config)

        raise ValueError(
            f"Unsupported chunking strategy: {config.chunking_strategy_type}")
