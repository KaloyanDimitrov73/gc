from typing import Literal
from knowledge_base.config.knowledge_graph_config import KnowledgeGraphConfig
from language_model.config.llm_config import LLMConfig
from retrieval.config.retrieval_config import RetrievalConfig


class KGRetrievalConfig(RetrievalConfig):
    """
    Configuration for retrievers that are of knowledge graph retrieval type.
    """
    type: Literal["kg_retrieval"] = "kg_retrieval"
    retriever_type: str
    index_llm_config: LLMConfig
    query_llm_config: LLMConfig
    knowledge_graph_config: KnowledgeGraphConfig

    def generate_name(self):
        return (
            f"{self.retriever_type}_"
            f"{self.index_llm_config.generate_name()}_"
            f"{self.query_llm_config.generate_name()}_"
            f"{self.knowledge_graph_config.generate_name()}"
        )
