from experimentation.config.evaluator_config import EvaluatorConfig
from core import ConfigurationManager


class EvaluatorConfigManager(ConfigurationManager[EvaluatorConfig]):
    """Class responsible for managing configurations of evaluator classes."""
    
    DEFAULT_FILE_NAME = "default_evaluators.json"
    DEFAULT_ROOT_NAME = "evaluators"
    CONFIG_CLASS = EvaluatorConfig
