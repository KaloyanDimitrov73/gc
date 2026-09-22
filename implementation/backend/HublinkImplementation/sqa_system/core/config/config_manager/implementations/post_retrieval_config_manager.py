from sqa_system.core.config.models import PostRetrievalConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager


class PostRetrievalConfigManager(ConfigurationManager[PostRetrievalConfig]):
    """Class responsible for managing post retrieval processing configurations."""

    DEFAULT_FILE_NAME = "default_post_retrieval_configs.json"
    DEFAULT_ROOT_NAME = "post_retrieval_pipes"
    CONFIG_CLASS = PostRetrievalConfig
