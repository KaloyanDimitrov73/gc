from sqa_system.core.config.models.pipeline_config import PipelineConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager


class PipelineConfigManager(ConfigurationManager[PipelineConfig]):
    """Class responsible for managing pipeline configurations."""

    DEFAULT_FILE_NAME = "default_pipelines.json"
    DEFAULT_ROOT_NAME = "pipelines"
    CONFIG_CLASS = PipelineConfig
