from sqa_system.core.config.models import PreRetrievalConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager


class PreRetrievalConfigManager(ConfigurationManager[PreRetrievalConfig]):
    """Class responsible for managing pre retrieval processing configurations."""

    DEFAULT_FILE_NAME = "default_pre_retrieval_configs.json"
    DEFAULT_ROOT_NAME = "pre_retrieval_pipes"
    CONFIG_CLASS = PreRetrievalConfig
