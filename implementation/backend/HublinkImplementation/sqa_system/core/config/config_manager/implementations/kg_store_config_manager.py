from sqa_system.core.config.models.knowledge_base.knowledge_graph_config import KnowledgeGraphConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager


class KGStoreConfigManager(ConfigurationManager[KnowledgeGraphConfig]):
    """Class responsible for managing the knowledge graph store configurations."""
 
    DEFAULT_FILE_NAME = "default_knowledge_graphs.json"
    DEFAULT_ROOT_NAME = "knowledge_graphs"
    CONFIG_CLASS = KnowledgeGraphConfig
