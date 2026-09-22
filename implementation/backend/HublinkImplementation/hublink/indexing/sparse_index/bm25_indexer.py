import bm25s
from typing import List

from core.logging.logging import get_logger
from hublink.core.sparse_index.sparse_storage_manager import (
    SparseStorageManager,
)
from knowledge_base.sparse_index_store.bm25_tokenizer import (
    tokenize_for_bm25,
)
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexArtifact,
    SparseIndexDocument,
)

logger = get_logger(__name__)


class Bm25Indexer:
    """Builds a persisted BM25 index from sparse path documents."""

    def __init__(self, storage_manager: SparseStorageManager) -> None:
        self.storage_manager = storage_manager

    def run_indexing(self, documents: List[SparseIndexDocument]) -> bool:
        if not documents:
            logger.debug("No documents supplied; keeping the existing BM25 index.")
            return False

        tokenized_corpus = tokenize_for_bm25(
            [document.text for document in documents]
        )
        nonempty_rows = [
            (document.metadata, tokens)
            for document, tokens in zip(documents, tokenized_corpus)
            if tokens
        ]
        if not nonempty_rows:
            logger.info("No tokenizable documents found; BM25 index was not built.")
            return self.storage_manager.store_bm25(
                SparseIndexArtifact(index=None, records=[])
            )

        records = [metadata for metadata, _ in nonempty_rows]
        tokens = [row_tokens for _, row_tokens in nonempty_rows]
        index = bm25s.BM25(method="lucene", corpus=records)
        index.index(tokens, show_progress=False)
        return self.storage_manager.store_bm25(SparseIndexArtifact(
            index=index,
            records=records,
        ))
