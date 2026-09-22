from typing import List

from torch import Tensor

from core.logging.logging import get_logger
from knowledge_base.sparse_index_store.splade_encoder import (
    DEFAULT_SPLADE_MODEL,
    get_splade_encoder,
)
from knowledge_base.sparse_index_store.sparse_index_cache import (
    load_sparse_index,
    save_sparse_index,
    splade_index_path,
)
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexArtifact,
    SparseIndexHit,
    SparseIndexStore,
)

logger = get_logger(__name__)


class SpladeSparseIndexStore(SparseIndexStore):
    """File-backed sparse index store using a SPLADE sparse tensor."""

    def __init__(
            self,
            index_key: str,
            model_name: str = DEFAULT_SPLADE_MODEL,
            load_index: bool = True) -> None:
        self._index_key = index_key
        self._configured_model_name = model_name
        self._model_name = model_name
        self._matrix: Tensor | None = None
        self._records: List[dict[str, str]] = []
        if load_index:
            self.reload()

    @property
    def name(self) -> str:
        return "splade"

    @property
    def index_key(self) -> str:
        return self._index_key

    @property
    def is_available(self) -> bool:
        return self._matrix is not None

    def store(self, artifact: SparseIndexArtifact) -> bool:
        matrix = artifact.index
        if not isinstance(matrix, Tensor) or not matrix.is_sparse:
            raise TypeError(
                "SPLADE store requires a sparse PyTorch tensor artifact."
            )
        model_name = artifact.metadata.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("SPLADE artifact requires a model name.")

        file_path = splade_index_path(self.index_key)
        save_sparse_index(
            file_path,
            {
                "records": artifact.records,
                # Keep the aligned legacy fields so existing caches and
                # diagnostics remain readable during this transition.
                "hub_ids": [record["hub_id"] for record in artifact.records],
                "path_hashes": [
                    record["path_hash"] for record in artifact.records
                ],
                "matrix": matrix,
                "model_name": model_name,
            },
        )
        logger.info("Persisted SPLADE index to %s", file_path)
        return self.reload()

    def search(
            self,
            query_text: str,
            limit: int | None = None) -> List[SparseIndexHit]:
        if self._matrix is None:
            return []

        encoder = get_splade_encoder(self._model_name)
        query_vector = encoder.encode_query(
            [query_text],
            batch_size=1,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )
        if isinstance(query_vector, list):
            raise TypeError(
                "SPLADE query encoding returned a list despite "
                "convert_to_tensor=True."
            )

        scores = (
            encoder.similarity(query_vector, self._matrix)
            .squeeze(0)
            .detach()
            .cpu()
            .tolist()
        )
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
        self._matrix = None
        self._records = []
        data = load_sparse_index(splade_index_path(self.index_key))
        if data is None:
            return False

        required_keys = {"hub_ids", "path_hashes", "matrix", "model_name"}
        if not required_keys.issubset(data):
            logger.error("SPLADE index has an unsupported payload format.")
            return False

        matrix = data["matrix"]
        if not isinstance(matrix, Tensor) or not matrix.is_sparse:
            logger.error("SPLADE index does not contain a sparse PyTorch tensor.")
            return False

        model_name = data["model_name"]
        if not isinstance(model_name, str) or not model_name:
            logger.error("SPLADE index has no valid model name.")
            return False

        records = data.get("records")
        if records is None:
            records = [
                {
                    "record_id": f"{hub_id}:{path_hash}",
                    "hub_id": hub_id,
                    "path_hash": path_hash,
                }
                for hub_id, path_hash in zip(
                    data["hub_ids"], data["path_hashes"]
                )
            ]

        if len(records) != matrix.shape[0]:
            logger.error(
                "SPLADE metadata has %d rows but its matrix has %d rows.",
                len(records),
                matrix.shape[0],
            )
            return False
        try:
            for record in records:
                record["hub_id"]
                record["path_hash"]
        except (KeyError, TypeError) as error:
            logger.error("Invalid SPLADE record metadata: %s", error)
            return False

        self._matrix = matrix
        self._records = records
        self._model_name = model_name
        return True
