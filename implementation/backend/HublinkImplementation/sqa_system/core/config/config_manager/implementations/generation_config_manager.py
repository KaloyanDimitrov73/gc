from sqa_system.core.config.models.pipe.generation_config import GenerationConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class GenerationConfigManager(ConfigurationManager[GenerationConfig]):
    """Class responsible for managing generation pipe configurations."""

    DEFAULT_FILE_NAME = "default_generation_pipes.json"
    DEFAULT_ROOT_NAME = "generation_pipes"
    CONFIG_CLASS = GenerationConfig
