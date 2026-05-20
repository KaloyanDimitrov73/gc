from experimentation.config.experiment_config import ExperimentConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager


class ExperimentConfigManager(ConfigurationManager[ExperimentConfig]):
    """Class responsible for managing experimentation configurations."""
    
    DEFAULT_FILE_NAME = "default_experiments.json"
    DEFAULT_ROOT_NAME = "experiments"
    CONFIG_CLASS = ExperimentConfig
