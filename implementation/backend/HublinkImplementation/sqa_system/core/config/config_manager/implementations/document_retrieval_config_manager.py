from sqa_system.core.config.models import DocumentRetrievalConfig
from sqa_system.core.config.config_manager.base.configuration_manager import ConfigurationManager

class DocumentRetrievalConfigManager(ConfigurationManager[DocumentRetrievalConfig]):
    """Class responsible for managing the knowledge graph retriever configurations."""
    
    DEFAULT_FILE_NAME = "default_document_retriever_pipes.json"
    DEFAULT_ROOT_NAME = "document_retrievers"
    CONFIG_CLASS = DocumentRetrievalConfig
