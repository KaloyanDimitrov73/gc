from sqa_system.core.config.models.embedding_config import EmbeddingConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager


class EmbeddingConfigManager(ConfigurationManager[EmbeddingConfig]):
    """Class responsible for managing embedding configurations."""
    
    DEFAULT_FILE_NAME = "default_embeddings.json"
    DEFAULT_ROOT_NAME = "embeddings"
    CONFIG_CLASS = EmbeddingConfig
