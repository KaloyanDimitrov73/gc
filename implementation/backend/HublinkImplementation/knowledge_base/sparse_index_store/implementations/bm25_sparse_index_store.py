import shutil
from pathlib import Path
from typing import Any, List

import bm25s

from core.logging.logging import get_logger
from knowledge_base.sparse_index_store.bm25_tokenizer import (
    tokenize_for_bm25,
)
from knowledge_base.sparse_index_store.sparse_index_cache import bm25_index_path
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexArtifact,
    SparseIndexHit,
    SparseIndexStore,
)

logger = get_logger(__name__)

class Bm25SparseIndexStore(SparseIndexStore):
    """File-backed sparse index store implemented with BM25S."""

    def __init__(self, index_key: str, load_index: bool = True) -> None:
        self._index_key = index_key
        self._bm25: bm25s.BM25 | None = None
        self._records: List[dict[str, str]] = []
        if load_index:
            self.reload()

    @property
    def name(self) -> str:
        return "bm25"

    @property
    def index_key(self) -> str:
        return self._index_key

    @property
    def is_available(self) -> bool:
        return self._bm25 is not None

    def store(self, artifact: SparseIndexArtifact) -> bool:
        index_path = Path(bm25_index_path(self.index_key))
        if index_path.is_dir():
            shutil.rmtree(index_path)
        elif index_path.exists():
            index_path.unlink()

        if artifact.index is None and not artifact.records:
            self._bm25 = None
            self._records = []
            return False
        if not isinstance(artifact.index, bm25s.BM25):
            raise TypeError("BM25 store requires a bm25s.BM25 index artifact.")

        artifact.index.save(
            index_path,
            corpus=artifact.records,
            show_progress=False,
        )
        logger.info("Persisted BM25S index to %s", index_path)
        return self.reload()

    def search(
            self,
            query_text: str,
            limit: int | None = None) -> List[SparseIndexHit]:
        if self._bm25 is None:
            return []

        query_tokens = tokenize_for_bm25([query_text])[0]
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        hits = [
            SparseIndexHit(
                record_id=record.get(
                    "record_id",
                    f"{record['hub_id']}:{record['path_hash']}",
                ),
                hub_id=record["hub_id"],
                path_hash=record["path_hash"],
                score=float(score),
            )
            for record, score in zip(self._records, scores)
            if float(score) > 0
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits if limit is None else hits[:limit]

    def reload(self) -> bool:
        self._bm25 = None
        self._records = []
        index_path = Path(bm25_index_path(self.index_key))
        if not index_path.is_dir():
            return False

        try:
            bm25 = bm25s.BM25.load(
                index_path,
                load_corpus=True,
                mmap=False,
                show_progress=False,
            )
        except Exception as error:
            logger.error(
                "Failed to load BM25S index from %s: %s. Re-run sparse indexing.",
                index_path,
                error,
            )
            return False

        if bm25.corpus is None:
            logger.error("BM25S index at %s has no metadata.", index_path)
            return False

        try:
            records: List[Any] = list(bm25.corpus)
            for record in records:
                record["hub_id"]
                record["path_hash"]
        except (KeyError, TypeError) as error:
            logger.error("Invalid BM25S metadata at %s: %s", index_path, error)
            return False

        document_count = bm25.scores["num_docs"]
        if len(records) != document_count:
            logger.error(
                "BM25S metadata has %d rows but its index has %d documents.",
                len(records),
                document_count,
            )
            return False

        self._bm25 = bm25
        self._records = records
        return True
