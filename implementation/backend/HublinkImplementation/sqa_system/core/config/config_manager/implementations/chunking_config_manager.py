from sqa_system.core.config.models.chunking_strategy_config import ChunkingStrategyConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager

class ChunkingConfigManager(ConfigurationManager[ChunkingStrategyConfig]):
    """Class responsible for managing the chunking strategy configurations."""
    
    DEFAULT_FILE_NAME = "default_chunking_strategies.json"
    DEFAULT_ROOT_NAME = "chunking_strategies"
    CONFIG_CLASS = ChunkingStrategyConfig
