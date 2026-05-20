from typing import Literal
from sqa_system.core.config.models.knowledge_base.knowledge_graph_config import KnowledgeGraphConfig
from sqa_system.core.config.models.llm_config import LLMConfig
from .retrieval_config import RetrievalConfig


class KGRetrievalConfig(RetrievalConfig):
    """
    Configuration for retrievers that are of knowledge graph retrieval type.
    """
    type: Literal["kg_retrieval"] = "kg_retrieval"
    retriever_type: str
    llm_config: LLMConfig
    knowledge_graph_config: KnowledgeGraphConfig

    def generate_name(self):
        return (
            f"{self.retriever_type}_"
            f"{self.llm_config.generate_name()}_"
            f"{self.knowledge_graph_config.generate_name()}"
        )
