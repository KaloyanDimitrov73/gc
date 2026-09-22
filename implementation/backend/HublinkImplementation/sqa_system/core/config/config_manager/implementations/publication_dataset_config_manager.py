from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class PublicationDatasetConfigManager(ConfigurationManager[DatasetConfig]):
    """Class responsible for managing publication dataset configurations."""

    DEFAULT_FILE_NAME = "default_publication_datasets.json"
    DEFAULT_ROOT_NAME = "publication_datasets"
    CONFIG_CLASS = DatasetConfig
