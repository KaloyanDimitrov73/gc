from sqa_system.core.config.models import VectorStoreConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager


class VectorStoreConfigManager(ConfigurationManager[VectorStoreConfig]):
    """Class responsible for managing vector store configurations."""

    DEFAULT_FILE_NAME = "default_vector_stores.json"
    DEFAULT_ROOT_NAME = "vector_stores"
    CONFIG_CLASS = VectorStoreConfig
