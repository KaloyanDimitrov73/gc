import asyncio

import weave
from typing_extensions import override

from backend.app.modules.qa.application import RetrievalService
from backend.app.modules.qa.infrastructure.hublink.hublink_service import HubLinkService
from core.data.models import RetrievalAnswer
from backend.app.modules.indexing.infrastructure.hublink.hub_store_service import (
    HubStoreService,
)
from hublink.indexing.experiment.sparse_store_service import (
    SparseStoreService,
)
from hublink.retrieval.experiment.hub_link_retriever_for_experiment import (
    HubLinkRetrieverForExperiment,
)
from knowledge_base.knowledge_graph.storage.implementations.orkg_remote_graph import ORKGRemoteGraph
from implementation.config.config_models.retrieval.kg_retrieval_config import KGRetrievalConfig
from implementation.pipe.retrieval.base.retrieval_pipe import RetrievalPipe
from implementation.shared_models.pipe_io_data import PipeIOData
from implementation.core.logging import get_logger

logger = get_logger(__name__)


class KGRetrievalWithHistoryPipe(RetrievalPipe[KGRetrievalConfig]):
    """
    A 'RetrievalPipe' class implementation that is responsible for retrieving data
    from a Knowledge Graph.
    
    Args:
        config (KGRetrievalConfig): The configuration for the Knowledge Graph retrieval.
            It contains the retriever configuration and other parameters.
    """

    def __init__(self, config: KGRetrievalConfig):
        super().__init__(config)
        self._prepare()

    @weave.op()
    @override
    def _process(self, input_data: PipeIOData) -> PipeIOData:
        """
        Uses the prepared Knowledge Graph retriever to retrieve
        contexts based on the initial question provided in the input data.

        The retrieved contexts are then appended to the input data.
        
        Args:
            input_data (PipeIOData): The input data that will be processed.
                It contains the initial question and other relevant information.
                
        Returns:
            PipeIOData: The processed input data with the retrieved contexts
                appended to it.
        """
        retrieval_answer = None
        contexts = None
        print("input")
        print(input_data.initial_question)
        print(input_data.message_history)
        try:
            retrieval_answer, contexts = asyncio.run(self._get_retrieval_answer(input_data))
        except Exception as e:
            logger.error(
                "Unhandled error in experiment streaming: %s", e, exc_info=True
            )

        if not retrieval_answer:
            logger.warning("No retrieval answer found.")
            return input_data

        input_data.generated_answer = retrieval_answer
        #input_data.retrieved_context = contexts
        return input_data


    async def _get_retrieval_answer(self, input_data: PipeIOData):
        retrieval_answer = None
        nodes = None
        contexts = None
        async for event in self.retrieval_service.ask_streaming_with_meta_router(
                question=input_data.retrieval_question,
                retrieval_mode="direct",
                llm_model="o3-mini",
                number_of_hubs=10,
                conversation_history=input_data.message_history
        ):
            if event.get("type") == "complete":
                retrieval_answer = event["answer"]
                nodes = event["nodes"]
        print("Answer")
        print(retrieval_answer)
        print(nodes)
        print(contexts)
        return retrieval_answer, contexts

    def _prepare(self):
        """
        Prepares the Knowledge Graph retriever for use in the retrieval process.
        This method initializes the retriever based on the provided configuration.
        """
        graph = ORKGRemoteGraph(self.config.knowledge_graph_config)
        graph.update_cache_if_not_exists()
        self.hub_store_service = HubStoreService(self.config)

        self.hublink = HubLinkService(graph=graph, hub_storage_manager=self.hub_store_service.hub_storage_manager)

        self.retrieval_service = RetrievalService(
            hublink_service=self.hublink,
            guardrails_service=None
        )

