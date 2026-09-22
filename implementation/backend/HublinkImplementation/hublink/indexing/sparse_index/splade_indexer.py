from typing import List

from hublink.core.sparse_index.sparse_storage_manager import (
    SparseStorageManager,
)
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexArtifact,
    SparseIndexDocument,
)
from knowledge_base.sparse_index_store.splade_encoder import (
    DEFAULT_SPLADE_MODEL,
    get_splade_encoder,
)


class SpladeIndexer:
    """Builds a persisted SPLADE index from sparse path documents."""

    def __init__(
            self,
            storage_manager: SparseStorageManager,
            model_name: str = DEFAULT_SPLADE_MODEL) -> None:
        self.storage_manager = storage_manager
        self.model_name = model_name

    def run_indexing(self, documents: List[SparseIndexDocument]) -> bool:
        if not documents:
            return False

        matrix = get_splade_encoder(self.model_name).encode_document(
            [document.text for document in documents],
            batch_size=16,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )
        if isinstance(matrix, list):
            raise TypeError(
                "SPLADE document encoding returned a list despite "
                "convert_to_tensor=True."
            )
        return self.storage_manager.store_splade(SparseIndexArtifact(
            index=matrix,
            records=[document.metadata for document in documents],
            metadata={"model_name": self.model_name},
        ))
