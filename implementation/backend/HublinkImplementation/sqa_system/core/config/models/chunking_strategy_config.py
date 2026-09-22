from typing_extensions import Annotated
from pydantic import Field
from sqa_system.core.config.models.base.config import Config


class ChunkingStrategyConfig(Config):
    """Configuration for a chunker class."""
    chunking_strategy_type: str
    chunk_size: Annotated[int, Field(ge=1)]
    chunk_overlap: Annotated[int, Field(ge=0)]

    def generate_name(self) -> str:
        return (f"{self.chunking_strategy_type.lower()}_csize{self.chunk_size}_"
                f"coverlap{self.chunk_overlap}")
