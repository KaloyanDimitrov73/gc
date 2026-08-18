from enum import Enum

from hublink.core.models.hub_link_settings import HubLinkSettings
from knowledge_base.vector_store.storage.vector_store import VectorStore
from language_model.config.llm_config import LLMConfig
from retrieval.config.kg_retrieval_config import KGRetrievalConfig


class VectorStoreType(Enum):
    """
    The following enum stores all the vector store that the system supports.
    """
    CHROMA = "chroma"

    # PINECONE = "pinecone"  # add here once implemented

    @staticmethod
    def get_values() -> list:
        """Returns a list of all the values in the enum."""
        return [backend.value for backend in VectorStoreType]

class VectorStoreProvider:
    """
    Returns a ``VectorStore`` implementation for the given backend name, and
    derives the deterministic store name from the relevant config objects.
    Add new backends as additional ``if`` branches — callers are unaffected.
    """

    @staticmethod
    def compute_store_name(
            config: KGRetrievalConfig,
            settings: HubLinkSettings,
            indexing_llm_config: LLMConfig | None = None) -> str:
        """
        Deterministic store name derived from the graph/embedding/indexing-LLM
        configs. Both the indexing and retrieval side MUST call this with
        equivalent configs — otherwise they end up pointing at two different
        stores.
        """

        graph_hash = config.knowledge_graph_config.config_hash
        if indexing_llm_config is None:
            indexing_llm_config = config.llm_config
        llm_hash = indexing_llm_config.config_hash
        embedding_hash = settings.embedding_config.config_hash

        return (
            f"{graph_hash}_"
            f"{embedding_hash}{llm_hash}"
        )

    @staticmethod
    def get_vector_store(backend: str, store_name: str, distance_metric: str) -> VectorStore:
        if backend == VectorStoreType.CHROMA.value:
            from knowledge_base.vector_store.storage.implementations.chroma_vector_store import (
                ChromaVectorStore,
            )
            return ChromaVectorStore(store_name=store_name, distance_metric=distance_metric)

        # Example for a future backend:

        raise ValueError(
            f"Unsupported vector store backend '{backend}'. "
            f"Supported backends: 'chroma'."
        )
