from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class QADatasetConfigManager(ConfigurationManager[DatasetConfig]):
    """Class responsible for managing QA dataset configurations."""

    DEFAULT_FILE_NAME = "default_qa_datasets.json"
    DEFAULT_ROOT_NAME = "qa_datasets"
    CONFIG_CLASS = DatasetConfig
