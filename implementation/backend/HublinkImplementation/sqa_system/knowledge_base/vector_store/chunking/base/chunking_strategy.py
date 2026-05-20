from abc import ABC, abstractmethod
from sqa_system.core.config.models.chunking_strategy_config import ChunkingStrategyConfig
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.data.models.context import Context


class ChunkingStrategy(ABC):
    """An interface for a strategy that creates chunks for a given publication."""

    def __init__(self, config: ChunkingStrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def create_chunks(self, publication: Publication) -> list[Context]:
        """
        Creates chunks for the given publication.

        Args:
            publication (Publication): The publication to create chunks for.

        Returns:
            list[Context]: A list of Context objects representing the chunks.
        """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Returns the name of the chunking strategy.

        Returns:
            str: The name of the chunking strategy.
        """
