from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager
from sqa_system.core.config.models.llm_config import LLMConfig


class LLMConfigManager(ConfigurationManager[LLMConfig]):
    """Class responsible for managing LLM configurations."""

    DEFAULT_FILE_NAME = "default_llms.json"
    DEFAULT_ROOT_NAME = "llms"
    CONFIG_CLASS = LLMConfig
