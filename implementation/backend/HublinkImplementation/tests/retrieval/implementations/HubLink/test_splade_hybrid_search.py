from unittest.mock import MagicMock, call, patch

import torch

from hublink.core.models.hub_path import HubPath
from hublink.core.sparse_index.sparse_search_result import SparseSearchResult
from hublink.core.sparse_index.sparse_storage_manager import SparseStorageManager
from hublink.indexing.sparse_index.splade_indexer import SpladeIndexer
from hublink.retrieval.candidate_hub_finder.hybrid_search.splade_fusion_decorator import (
    SpladeFusionDecorator,
)
from hublink.retrieval.models.processed_question import ProcessedQuestion
from knowledge_base.sparse_index_store import (
    splade_encoder as splade_encoder_module,
)
from knowledge_base.sparse_index_store.implementations.splade_sparse_index_store import (
    SpladeSparseIndexStore,
)
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexDocument,
)


def _path(path_hash: str, score: float) -> HubPath:
    return HubPath(
        path_text=f"path {path_hash}",
        path_hash=path_hash,
        path=[],
        dense_score=score,
        score=score,
    )


def test_splade_decorator_skips_sparse_search_without_keywords():
    dense_hubs = {"dense-hub": [_path("dense-path", 0.9)]}
    wrapped_finder = MagicMock()
    wrapped_finder.find_candidate_hubs.return_value = dense_hubs
    sparse_storage_manager = MagicMock()
    decorator = SpladeFusionDecorator(
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
    sparse_storage_manager.search_splade.assert_not_called()


def test_splade_decorator_searches_full_question_with_keywords():
    dense_hubs = {"dense-hub": [_path("dense-path", 0.9)]}
    wrapped_finder = MagicMock()
    wrapped_finder.find_candidate_hubs.return_value = dense_hubs
    sparse_storage_manager = MagicMock()
    sparse_storage_manager.search_splade.return_value = SparseSearchResult(
        ranked_hub_ids=[],
        hits_by_hub={},
    )
    decorator = SpladeFusionDecorator(
        candidate_hub_finder=wrapped_finder,
        hub_storage_manager=MagicMock(),
        sparse_storage_manager=sparse_storage_manager,
        top_paths_to_keep=2,
        number_of_hubs=2,
    )

    result = decorator.find_candidate_hubs(ProcessedQuestion(
        question="papers by Georg Buchgeher",
        keywords=["Georg Buchgeher"],
        embeddings=[[1.0, 0.0]],
    ))

    assert result == dense_hubs
    sparse_storage_manager.search_splade.assert_called_once_with(
        query_text="papers by Georg Buchgeher",
        top_hubs=2,
        paths_per_hub=2,
    )


def test_splade_encoder_is_cached_and_forced_to_cpu():
    splade_encoder_module._encoder_cache.clear()
    encoder = MagicMock()

    try:
        with patch.object(
            splade_encoder_module,
            "SparseEncoder",
            return_value=encoder,
        ) as sparse_encoder_class:
            first = splade_encoder_module.get_splade_encoder(
                "naver/test-splade"
            )
            second = splade_encoder_module.get_splade_encoder(
                "naver/test-splade"
            )

        assert first is encoder
        assert second is encoder
        sparse_encoder_class.assert_called_once_with(
            "naver/test-splade",
            device="cpu",
        )
    finally:
        splade_encoder_module._encoder_cache.clear()


def test_splade_store_uses_document_encoding_and_records_model():
    documents = [
        SparseIndexDocument(
            record_id="hub-a:path-1", text="first path",
            hub_id="hub-a", path_hash="path-1",
        ),
        SparseIndexDocument(
            record_id="hub-a:path-2", text="second path",
            hub_id="hub-a", path_hash="path-2",
        ),
    ]
    matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [1]]),
        values=torch.tensor([1.0]),
        size=(2, 3),
    )
    encoder = MagicMock()
    encoder.encode_document.return_value = matrix

    with patch(
        "hublink.indexing.sparse_index.splade_indexer.get_splade_encoder",
        return_value=encoder,
    ), patch(
        "knowledge_base.sparse_index_store.implementations."
        "splade_sparse_index_store."
        "splade_index_path",
        return_value="splade-path.pkl",
    ), patch.object(
        SpladeSparseIndexStore,
        "reload",
        return_value=True,
    ):
        with patch(
            "knowledge_base.sparse_index_store.implementations."
            "splade_sparse_index_store."
            "save_sparse_index"
        ) as save_index:
            store = SpladeSparseIndexStore(
                "index-key", load_index=False
            )
            manager = SparseStorageManager(
                "index-key",
                splade_store=store,
            )
            result = SpladeIndexer(manager).run_indexing(documents)

    assert result
    assert encoder.encode_document.call_args_list == [
        call(["first path", "second path"],
             batch_size=16,
             convert_to_tensor=True,
             convert_to_sparse_tensor=True,
             save_to_cpu=True,
             ),
    ]
    path_payload = save_index.call_args_list[0].args[1]
    assert path_payload["hub_ids"] == ["hub-a", "hub-a"]
    assert path_payload["path_hashes"] == ["path-1", "path-2"]
    assert path_payload["matrix"] is matrix
    assert path_payload["model_name"]


def test_sparse_storage_manager_uses_splade_query_encoding_and_similarity():
    matrix = MagicMock()
    matrix.shape = (3, 30_522)
    query_embedding = MagicMock()
    similarity_scores = MagicMock()
    similarity_scores.squeeze.return_value.detach.return_value.cpu.return_value.tolist.return_value = [
        0.3,
        1.2,
        0.8,
    ]
    encoder = MagicMock()
    encoder.encode_query.return_value = query_embedding
    encoder.similarity.return_value = similarity_scores
    store = SpladeSparseIndexStore("index-key", load_index=False)
    store._records = [
        {"hub_id": "hub-a", "path_hash": "path-1"},
        {"hub_id": "hub-b", "path_hash": "path-2"},
        {"hub_id": "hub-a", "path_hash": "path-3"},
    ]
    store._matrix = matrix
    store._model_name = "naver/test-splade"
    manager = SparseStorageManager("index-key", splade_store=store)

    with patch(
        "knowledge_base.sparse_index_store.implementations."
        "splade_sparse_index_store.get_splade_encoder",
        return_value=encoder,
    ) as get_encoder:
        result = manager.search_splade(
            query_text="test query",
            top_hubs=2,
            paths_per_hub=2,
        )

    get_encoder.assert_called_once_with("naver/test-splade")
    encoder.encode_query.assert_called_once_with(
        ["test query"],
        batch_size=1,
        convert_to_tensor=True,
        convert_to_sparse_tensor=True,
        save_to_cpu=True,
    )
    encoder.similarity.assert_called_once_with(query_embedding, matrix)
    assert result.ranked_hub_ids == ["hub-b", "hub-a"]
    assert [
        hit.path_hash for hit in result.hits_by_hub["hub-a"]
    ] == ["path-3", "path-1"]


def test_splade_store_rejects_legacy_scipy_index_payload():
    legacy_payload = {
        "hub_ids": ["hub-a"],
        "path_hashes": ["path-1"],
        "matrix": MagicMock(),
    }

    with patch(
        "knowledge_base.sparse_index_store.implementations."
        "splade_sparse_index_store.splade_index_path",
        return_value="splade.pkl",
    ):
        with patch(
            "knowledge_base.sparse_index_store.implementations."
            "splade_sparse_index_store.load_sparse_index",
            return_value=legacy_payload,
        ):
            store = SpladeSparseIndexStore("index-key")
            assert not store.is_available


def test_splade_store_loads_current_tensor_index_payload():
    matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [1]]),
        values=torch.tensor([1.0]),
        size=(1, 3),
    )
    payload = {
        "hub_ids": ["hub-a"],
        "path_hashes": ["path-1"],
        "matrix": matrix,
        "model_name": "naver/test-splade",
    }

    with patch(
        "knowledge_base.sparse_index_store.implementations."
        "splade_sparse_index_store.splade_index_path",
        return_value="splade.pkl",
    ):
        with patch(
            "knowledge_base.sparse_index_store.implementations."
            "splade_sparse_index_store.load_sparse_index",
            return_value=payload,
        ):
            store = SpladeSparseIndexStore("index-key")

    assert store.is_available
    assert store._matrix is matrix
    assert store._model_name == "naver/test-splade"
