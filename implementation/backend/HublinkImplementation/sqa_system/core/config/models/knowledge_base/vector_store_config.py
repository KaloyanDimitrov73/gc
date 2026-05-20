from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.config.models.chunking_strategy_config import ChunkingStrategyConfig
from sqa_system.core.config.models.base.config import Config
from sqa_system.core.config.models.embedding_config import EmbeddingConfig


class VectorStoreConfig(Config):
    """Configuration class for a vector store."""
    vector_store_type: str
    chunking_strategy_config: ChunkingStrategyConfig
    embedding_config: EmbeddingConfig
    dataset_config: DatasetConfig
    force_index_rebuild: bool = False

    def generate_name(self) -> str:
        return (
            f"{self.vector_store_type.lower()}_"
            f"{self.chunking_strategy_config.name}_"
            f"{self.embedding_config.name}_{self.dataset_config.name}"
        )
