import weave
from typing_extensions import override

from core.data.models import RetrievalAnswer
from retrieval.factory.document_retriever_factory import DocumentRetrieverFactory
from retrieval.config.document_retrieval_config import DocumentRetrievalConfig
from implementation.pipe.retrieval.base.retrieval_pipe import RetrievalPipe
from implementation.shared_models.pipe_io_data import PipeIOData


class DocumentRetrievalPipe(RetrievalPipe[DocumentRetrievalConfig]):
    """
    A 'RetrievalPipe' class implementation that is responsible for retrieving data
    from Publications.
    
    Args:
        config (DocumentRetrievalConfig): The configuration for the document retrieval.
            It contains the retriever configuration and other parameters.
    """

    def __init__(self, config: DocumentRetrievalConfig):
        """
        Initializes the 'DocumentRetrievalPipe' object with the provided
        'DocumentRetrievalConfig' object.

        Args: 
            config (DocumentRetrievalConfig): The configuration object for 
                the retrieval process.
        """
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
        retrieval_answer: RetrievalAnswer = self.retriever.retrieve(
            query_text=input_data.retrieval_question
        )
        input_data.retrieved_context = input_data.retrieved_context or []
        input_data.retrieved_context.extend(retrieval_answer.contexts)
        if retrieval_answer.retriever_answer:
            input_data.generated_answer = retrieval_answer.retriever_answer
        return input_data

    def _prepare(self):
        kg_retriever_manager = DocumentRetrieverFactory()
        self.retriever = kg_retriever_manager.create(self.config)
