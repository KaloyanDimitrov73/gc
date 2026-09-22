from typing import List, Tuple
from typing_extensions import override
import io
import contextlib
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from sqa_system.core.logging.logging import get_logger
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.data.models.context import Context
from ..base.vector_store_adapter import VectorStoreAdapter

logger = get_logger(__name__)


class LangchainVectorStoreAdapter(VectorStoreAdapter):
    """Adapts the langchain vector stores."""

    ADDITIONAL_CONFIG_PARAMS = [
        AdditionalConfigParameter(
            name="distance_metric",
            description=("The distance metric to use. Available values are 'cosine'"
                         " for cosine similarity, 'l2' for squared L2, 'ip' for inner product."),
            param_type=str,
            available_values=['cosine', 'l2', 'ip'],
            default_value='l2'
        )
    ]

    @override
    def query(self, query_text: str, n_results: int) -> List[Context]:
        documents: List[Document] = self.vector_store.similarity_search(query=query_text,
                                                                        k=n_results)
        return self.convert_documents_to_contexts(documents)

    @override
    def query_with_metadata_filter(self,
                                   query_text: str,
                                   n_results: int,
                                   metadata_filter: dict) -> List[Context]:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Currently the method sometimes spams the console with errors
            # if the similarity score is negative. This however is no issue
            # for us. As we can not suppress the output of the method using
            # a parameter, we redirect the output to a string buffer.
            results: List[Tuple[Document, float]] = self.vector_store.similarity_search_with_relevance_scores(
                query=query_text,
                k=n_results,
                filter=metadata_filter
            )
        if len(results) == 0:
            logger.warning("No results found for query: %s", query_text)
            return []
        contexts = self.convert_documents_to_contexts(
            [doc for doc, _ in results])

        # add the scores
        for i, (_, score) in enumerate(results):
            contexts[i].score = score

        return contexts

    @override
    def get_retriever(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever()
