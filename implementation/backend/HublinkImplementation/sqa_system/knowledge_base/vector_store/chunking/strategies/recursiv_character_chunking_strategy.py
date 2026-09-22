from typing import List
from typing_extensions import override
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sqa_system.core.config.models.chunking_strategy_config import ChunkingStrategyConfig
from sqa_system.knowledge_base.vector_store.chunking.base.chunking_strategy import ChunkingStrategy
from sqa_system.core.data.models.context import Context, ContextType
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class RecursiveCharacterChunkingStrategy(ChunkingStrategy):
    """
    A chunking strategy that uses recursive character text splitter from langchain.
    """

    def __init__(self, config: ChunkingStrategyConfig) -> None:
        super().__init__(config)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size,
                                                            chunk_overlap=config.chunk_overlap)

    @classmethod
    @override
    def get_name(cls) -> str:
        return "RecursiveCharacterChunkingStrategy"

    @override
    def create_chunks(self, publication: Publication) -> List[Context]:
        chunks = []
        try:
            document = publication.to_document()
            if document is None:
                logger.error("Document must not be None.")
                raise ValueError("Document must not be None.")

            document_chunks = self.text_splitter.split_documents([document])
            if document_chunks is None:
                logger.error("Document chunks must not be None.")
                raise ValueError("Document chunks must not be None.")

            for document_chunk in document_chunks:
                if document_chunk is None:
                    continue

                try:
                    chunk = Context.from_document(document_chunk)
                    chunk.context_type = ContextType.DOC
                    if chunk is not None:
                        chunks.append(chunk)
                except Exception as e:
                    logger.error("Failed to create chunk from document: %s", e)
                    print(f"Failed to create chunk from document: {e}")

            return chunks
        except Exception as e:
            logger.error("Failed to create chunks for publication: %s", e)
            return []
