from sqa_system.core.config.models import KGRetrievalConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager


class KGRetrievalConfigManager(ConfigurationManager[KGRetrievalConfig]):
    """Class responsible for managing the knowledge graph retriever pipe configurations."""

    DEFAULT_FILE_NAME = "default_knowledge_graph_retriever_pipes.json"
    DEFAULT_ROOT_NAME = "knowledge_graph_retrievers"
    CONFIG_CLASS = KGRetrievalConfig
