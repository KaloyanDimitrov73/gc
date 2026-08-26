from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hublink.core.sparse_index.sparse_storage_manager import SparseStorageManager
from hublink.indexing.sparse_index.bm25_indexer import Bm25Indexer
from knowledge_base.sparse_index_store.implementations.bm25_sparse_index_store import (
    Bm25SparseIndexStore,
)
from knowledge_base.sparse_index_store.sparse_index_cache import (
    bm25_index_path,
)
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexDocument,
)
from hublink.retrieval.candidate_hub_finder.hybrid_search.bm25_fusion_decorator import (
    Bm25FusionDecorator,
)
from hublink.core.sparse_index.sparse_search_result import (
    SparsePathHit,
    SparseSearchResult,
)
from hublink.retrieval.models.processed_question import ProcessedQuestion


@pytest.mark.parametrize("index_key", ["", ".", "..", "nested/index"])
def test_bm25_index_path_rejects_unsafe_cache_keys(index_key: str):
    with pytest.raises(ValueError, match="non-empty file name"):
        bm25_index_path(index_key)


def test_bm25s_index_round_trip_preserves_path_metadata(tmp_path: Path):
    documents = [
        SparseIndexDocument(
            record_id="hub-a:path-a1", text="alpha first path",
            hub_id="hub-a", path_hash="path-a1",
        ),
        SparseIndexDocument(
            record_id="hub-a:path-a2", text="alpha second path",
            hub_id="hub-a", path_hash="path-a2",
        ),
        SparseIndexDocument(
            record_id="hub-b:path-b1", text="rare quasar 2042",
            hub_id="hub-b", path_hash="path-b1",
        ),
    ]
    index_root = tmp_path / "bm25s-index"
    index_path = index_root / "path"

    with patch(
        "knowledge_base.sparse_index_store.implementations."
        "bm25_sparse_index_store.bm25_index_path",
        return_value=str(index_path),
    ):
        store = Bm25SparseIndexStore("index-key", load_index=False)
        storage_manager = SparseStorageManager(
            "index-key",
            bm25_store=store,
        )
        assert Bm25Indexer(storage_manager).run_indexing(documents)

    assert (index_path / "params.index.json").is_file()
    assert (index_path / "corpus.jsonl").is_file()

    with patch(
        "knowledge_base.sparse_index_store.implementations."
        "bm25_sparse_index_store.bm25_index_path",
        return_value=str(index_path),
    ):
        store = Bm25SparseIndexStore("index-key")
        sparse_storage_manager = SparseStorageManager(
            "index-key",
            bm25_store=store,
        )

    assert sparse_storage_manager.has_bm25
    path_result = sparse_storage_manager.search_bm25(
        query_text="quasar 2042",
        top_hubs=2,
        paths_per_hub=2,
    )
    assert path_result.ranked_hub_ids == ["hub-b"]
    assert path_result.hits_by_hub["hub-b"][0].path_hash == "path-b1"


def test_bm25_store_removes_stale_index_when_no_text_is_tokenizable(
        tmp_path: Path):
    index_root = tmp_path / "bm25s-index"
    index_path = index_root / "path"
    index_path.mkdir(parents=True)
    (index_path / "stale-index-file").touch()
    documents = [SparseIndexDocument(
        record_id="hub-a:ignored",
        text="!!!",
        hub_id="hub-a",
        path_hash="ignored",
    )]

    with patch(
        "knowledge_base.sparse_index_store.implementations."
        "bm25_sparse_index_store.bm25_index_path",
        return_value=str(index_path),
    ):
        store = Bm25SparseIndexStore("index-key", load_index=False)
        storage_manager = SparseStorageManager(
            "index-key",
            bm25_store=store,
        )
        built = Bm25Indexer(storage_manager).run_indexing(documents)

    assert not built
    assert not index_path.exists()


def test_hybrid_decorator_always_merges_sparse_path_evidence():
    dense_path = MagicMock(path_hash="dense-path")
    ann_path = MagicMock(path_hash="ann-path")
    wrapped_finder = MagicMock()
    wrapped_finder.find_candidate_hubs.return_value = {
        "dense-hub": [dense_path]
    }
    sparse_storage_manager = MagicMock()
    sparse_storage_manager.search_bm25.return_value = SparseSearchResult(
        ranked_hub_ids=["sparse-hub"],
        hits_by_hub={
            "sparse-hub": [SparsePathHit(
                path_hash="sparse-path",
                hub_id="sparse-hub",
                score=2.0,
                rank=0,
            )]
        },
    )
    storage_manager = MagicMock()
    storage_manager.similarity_search_by_hub_entity.return_value = [ann_path]
    evidence_merger = MagicMock()
    evidence_merger.merge.side_effect = (
        lambda **kwargs: kwargs["existing_paths"]
    )
    decorator = Bm25FusionDecorator(
        candidate_hub_finder=wrapped_finder,
        hub_storage_manager=storage_manager,
        sparse_storage_manager=sparse_storage_manager,
        top_paths_to_keep=2,
        number_of_hubs=2,
        evidence_merger=evidence_merger,
    )

    result = decorator.find_candidate_hubs(ProcessedQuestion(
        question="rare query",
        keywords=["rare"],
        embeddings=[[1.0, 0.0]],
    ))

    assert result["dense-hub"] == [dense_path]
    assert result["sparse-hub"] == [ann_path]
    sparse_storage_manager.search_bm25.assert_called_once_with(
        query_text="rare query",
        top_hubs=2,
        paths_per_hub=2,
    )
    assert evidence_merger.merge.call_count == 2
    sparse_call = evidence_merger.merge.call_args_list[1]
    assert sparse_call.kwargs["existing_paths"] == [ann_path]
    assert sparse_call.kwargs["sparse_hits"] == (
        sparse_storage_manager.search_bm25.return_value
        .hits_by_hub["sparse-hub"]
    )
    assert sparse_call.kwargs["sparse_channel"] == "bm25"


def test_hybrid_candidate_union_does_not_rerank_hubs():
    dense_path = MagicMock(path_hash="dense-path")
    shared_path = MagicMock(path_hash="shared-path")
    sparse_only_path = MagicMock(path_hash="sparse-only-path")
    wrapped_finder = MagicMock()
    wrapped_finder.find_candidate_hubs.return_value = {
        "dense-hub": [dense_path],
        "shared-hub": [shared_path],
    }
    sparse_storage_manager = MagicMock()
    sparse_storage_manager.search_bm25.return_value = SparseSearchResult(
        ranked_hub_ids=["shared-hub", "sparse-only-hub"],
        hits_by_hub={},
    )
    storage_manager = MagicMock()
    storage_manager.similarity_search_by_hub_entity.return_value = [
        sparse_only_path
    ]
    decorator = Bm25FusionDecorator(
        candidate_hub_finder=wrapped_finder,
        hub_storage_manager=storage_manager,
        sparse_storage_manager=sparse_storage_manager,
        top_paths_to_keep=2,
        number_of_hubs=2,
    )

    result = decorator.find_candidate_hubs(ProcessedQuestion(
        question="query",
        keywords=["query"],
        embeddings=[[1.0, 0.0]],
    ))

    assert list(result) == [
        "dense-hub", "shared-hub", "sparse-only-hub"
    ]


def test_bm25_decorator_skips_sparse_search_without_keywords():
    dense_hubs = {"dense-hub": [MagicMock(path_hash="dense-path")]}
    wrapped_finder = MagicMock()
    wrapped_finder.find_candidate_hubs.return_value = dense_hubs
    sparse_storage_manager = MagicMock()
    decorator = Bm25FusionDecorator(
        candidate_hub_finder=wrapped_finder,
        hub_storage_manager=MagicMock(),
        sparse_storage_manager=sparse_storage_manager,
        top_paths_to_keep=2,
        number_of_hubs=2,
    )

    result = decorator.find_candidate_hubs(ProcessedQuestion(
        question="conceptual query",
        keywords=[],
        embeddings=[[1.0, 0.0]],
    ))

    assert result == dense_hubs
    wrapped_finder.find_candidate_hubs.assert_called_once()
    sparse_storage_manager.search_bm25.assert_not_called()
