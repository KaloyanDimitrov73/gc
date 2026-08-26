from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from hublink.core.hub_storage_manager import HubStorageManager
from hublink.core.models.hub_path import HubPath
from hublink.core.sparse_index.sparse_storage_manager import SparseStorageManager
from hublink.core.sparse_index.sparse_ranking import (
    build_sparse_search_result,
)
from hublink.core.sparse_index.sparse_search_result import (
    SparsePathHit,
    SparseSearchResult,
)
from hublink.indexing.sparse_index.splade_indexer import SpladeIndexer
from hublink.retrieval.candidate_hub_finder.ann_hub_finder import ANNHubFinder
from hublink.retrieval.candidate_hub_finder.hybrid_search.splade_fusion_decorator import (
    SpladeFusionDecorator,
)
from hublink.retrieval.candidate_hub_finder.hybrid_search.sparse_evidence_merger import (
    SparseEvidenceMerger,
)
from hublink.retrieval.models.processed_question import ProcessedQuestion
from hublink.retrieval.strategies.direct_retrieval_strategy import (
    DirectRetrievalStrategy,
)
from knowledge_base.vector_store.storage.vector_store import VectorScoreResults
from knowledge_base.sparse_index_store import splade_encoder as splade_encoder_module
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
        score=score
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


def test_sparse_ranking_preserves_path_hits_for_each_ranked_hub():
    result = build_sparse_search_result(
        scores=[0.2, 0.9, 0.8, 0.0],
        hub_ids=["hub-a", "hub-b", "hub-a", "hub-c"],
        path_hashes=["a-low", "b-high", "a-high", "c-zero"],
        top_hubs=2,
        paths_per_hub=2
    )

    assert result.ranked_hub_ids == ["hub-b", "hub-a"]
    assert [
        hit.path_hash for hit in result.hits_by_hub["hub-a"]
    ] == ["a-high", "a-low"]
    assert result.hits_by_hub["hub-a"][0].rank == 0
    assert result.hits_by_hub["hub-a"][0].global_rank == 1
    assert result.hits_by_hub["hub-a"][1].global_rank == 2


def test_sparse_ranking_rejects_misaligned_index_rows():
    with pytest.raises(ValueError, match="equal lengths"):
        build_sparse_search_result(
            scores=[1.0],
            hub_ids=["hub-a"],
            path_hashes=[],
            top_hubs=1,
            paths_per_hub=1
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


def test_sparse_evidence_merger_combines_dense_and_sparse_path_evidence():
    existing_shared = _path("shared", 0.91)
    sparse_only = HubPath(
        path_text="path sparse-only",
        path_hash="sparse-only",
        path=[],
    )
    storage_manager = MagicMock()
    storage_manager.retrieve_hub_paths_by_keys.return_value = {
        "sparse-only": sparse_only
    }
    merger = SparseEvidenceMerger(storage_manager)

    merged = merger.merge(
        existing_paths=[_path("dense-only", 0.80), existing_shared],
        sparse_hits=[
            SparsePathHit(
                path_hash="shared", hub_id="hub", score=3.0, rank=0),
            SparsePathHit(
                path_hash="sparse-only", hub_id="hub", score=2.0, rank=1)
        ]
    )

    assert [path.path_hash for path in merged] == [
        "dense-only", "shared", "sparse-only"
    ]
    assert merged[1] is existing_shared
    assert existing_shared.score == 0.91
    assert existing_shared.dense_score == 0.91
    assert existing_shared.sparse_scores == {"sparse": 3.0}
    assert existing_shared.sparse_ranks == {"sparse": 0}
    assert sparse_only.dense_score is None
    assert sparse_only.dense_rank is None
    assert sparse_only.sparse_scores == {"sparse": 2.0}
    assert sparse_only.sparse_ranks == {"sparse": 1}
    storage_manager.retrieve_hub_paths_by_keys.assert_called_once_with(
        hash_keys=["sparse-only"],
    )


def test_exact_sparse_path_is_rescored_with_dense_cosine_similarity():
    vector_store = MagicMock()
    vector_store.get_records_with_metadata_by_ids.return_value = (
        VectorScoreResults(
            ids=["path-1"],
            metadata=[{
                "path_hash": "path-1",
                "path": "serialized path",
                "path_text": "path text",
                "embedded_text": "path text"
            }],
            embeddings=[[1.0, 0.0]]
        )
    )
    manager = object.__new__(HubStorageManager)
    manager.vector_store = vector_store
    parsed_path = _path("path-1", 0.0)

    with patch(
        "hublink.core.hub_storage_manager.parse_hub_path",
        return_value=parsed_path
    ):
        result = manager.retrieve_scored_hub_paths_by_keys(
            path_hashes=["path-1"],
            query_embeddings=[[0.0, 1.0], [1.0, 0.0]]
        )

    assert result["path-1"].score == pytest.approx(1.0)
    assert result["path-1"].dense_score == pytest.approx(1.0)
    assert result["path-1"].embedded_text == "path text"


def test_dense_query_result_assigns_dense_and_final_path_score():
    manager = object.__new__(HubStorageManager)
    manager.diversity_penalty = 0
    result = VectorScoreResults(
        ids=["dense-record"],
        metadata=[{
            "path_hash": "path-1",
            "path": "serialized path",
            "path_text": "path text",
            "embedded_text": "path text",
        }],
        distances=[0.25],
    )

    with patch(
        "hublink.core.hub_storage_manager.parse_hub_path",
        return_value=_path("path-1", 0.0),
    ):
        paths = manager._process_query_results_to_hub_paths([result])

    assert paths[0].dense_score == pytest.approx(0.75)
    assert paths[0].score == pytest.approx(0.75)


def test_dense_diversity_penalty_keeps_dense_and_final_scores_equal():
    manager = object.__new__(HubStorageManager)
    manager.diversity_penalty = 0.1
    first = _path("first", 0.8)
    first.embedded_text = "(shared subject, first predicate, first object)"
    second = _path("second", 0.7)
    second.embedded_text = "(shared subject, second predicate, second object)"

    paths = manager._diversity_ranker_for_triples([first, second])

    assert paths[0].dense_score == paths[0].score == pytest.approx(0.8)
    assert paths[1].dense_score == paths[1].score == pytest.approx(0.6)


def test_post_union_path_processing_does_not_retrieve_more_paths():
    strategy = object.__new__(DirectRetrievalStrategy)
    strategy._get_hub_paths_for_hub = MagicMock(return_value=[
        _path("dense-new", 0.50)
    ])
    sparse_path = _path("sparse-hit", 0.85)
    dense_duplicate = _path("dense-duplicate", 0.60)

    result = strategy._fill_or_remove_paths(
        processed_question=ProcessedQuestion(
            question="question",
            embeddings=[[1.0, 0.0]]
        ),
        candidate_hubs={"hub": [sparse_path, dense_duplicate]},
        path_threshold=3
    )

    assert [path.path_hash for path in result["hub"]] == [
        "sparse-hit", "dense-duplicate"
    ]
    assert all(path.score == path.dense_score for path in result["hub"])
    strategy._get_hub_paths_for_hub.assert_not_called()


def test_path_rrf_uses_only_independent_channel_membership():
    strategy = object.__new__(DirectRetrievalStrategy)
    strategy.settings = SimpleNamespace(rrf_k=60)

    dense_path = _path("dense", 0.99)
    dense_path.dense_rank = 0
    shared_path = _path("shared", 0.80)
    shared_path.dense_rank = 1
    shared_path.sparse_scores["splade"] = 12.0
    shared_path.sparse_ranks["splade"] = 0
    sparse_path = HubPath(
        path_text="path sparse",
        path_hash="sparse",
        path=[],
        sparse_scores={"splade": 10.0},
        sparse_ranks={"splade": 1},
    )

    result = strategy._fill_or_remove_paths(
        processed_question=ProcessedQuestion(
            question="question",
            embeddings=[[1.0, 0.0]],
        ),
        candidate_hubs={
            "hub-a": [dense_path, shared_path, sparse_path],
        },
        path_threshold=2,
    )

    assert [path.path_hash for path in result["hub-a"]] == [
        "shared", "dense"
    ]
    best_per_channel = 2.0 / 61.0
    assert shared_path.score == pytest.approx(
        ((1.0 / 62.0) + (1.0 / 61.0)) / best_per_channel
    )
    assert dense_path.score == pytest.approx((1.0 / 61.0) / best_per_channel)
    assert sparse_path.score == pytest.approx((1.0 / 62.0) / best_per_channel)
    assert sparse_path.dense_score is None
    assert sparse_path.dense_rank is None


def test_path_rrf_uses_all_three_non_empty_channel_rankings():
    strategy = object.__new__(DirectRetrievalStrategy)
    strategy.settings = SimpleNamespace(rrf_k=60)

    shared_path = _path("shared", 0.90)
    shared_path.dense_rank = 0
    shared_path.sparse_ranks = {
        "bm25": 0,
        "splade": 0,
    }
    dense_only = _path("dense-only", 0.80)
    dense_only.dense_rank = 1
    bm25_only = HubPath(
        path_text="path bm25-only",
        path_hash="bm25-only",
        path=[],
        sparse_ranks={"bm25": 1},
    )
    splade_only = HubPath(
        path_text="path splade-only",
        path_hash="splade-only",
        path=[],
        sparse_ranks={"splade": 1},
    )

    result = strategy._score_and_limit_paths(
        candidate_hubs={
            "hub": [shared_path, dense_only, bm25_only, splade_only],
        },
        path_threshold=4,
    )

    assert result["hub"][0] is shared_path
    assert shared_path.score == pytest.approx(1.0)
    best_per_channel = 3.0 / 61.0
    expected_single_channel_score = (1.0 / 62.0) / best_per_channel
    assert dense_only.score == pytest.approx(expected_single_channel_score)
    assert bm25_only.score == pytest.approx(expected_single_channel_score)
    assert splade_only.score == pytest.approx(expected_single_channel_score)


def test_ann_finder_fills_dense_hubs_before_assigning_global_ranks():
    storage_manager = MagicMock()
    storage_manager.similarity_search_hubs.return_value = {
        "hub": [_path("dense-low", 0.80)]
    }
    storage_manager.similarity_search_by_hub_entity.return_value = [
        _path("dense-high", 0.90)
    ]
    finder = ANNHubFinder(
        hub_storage_manager=storage_manager,
        number_of_hubs=1,
        top_paths_to_keep=2,
    )
    question = ProcessedQuestion(
        question="question",
        embeddings=[[1.0, 0.0]],
    )

    result = finder.find_candidate_hubs(question)

    assert [path.path_hash for path in result["hub"]] == [
        "dense-high", "dense-low"
    ]
    assert [path.dense_rank for path in result["hub"]] == [0, 1]
    storage_manager.similarity_search_by_hub_entity.assert_called_once_with(
        query_embeddings=question.embeddings,
        hub_entity_id="hub",
        n_results=2,
        excluded_path_hashs=["dense-low"],
    )


def test_prune_hubs_aggregates_final_path_score():
    strategy = object.__new__(DirectRetrievalStrategy)
    strategy.settings = SimpleNamespace(number_of_hubs=1)
    hybrid_relevant = SimpleNamespace(
        paths=[HubPath(
            path_text="hybrid",
            path_hash="hybrid",
            path=[],
            dense_score=0.10,
            score=0.95,
        )],
        hub_score=None,
    )
    dense_relevant = SimpleNamespace(
        paths=[HubPath(
            path_text="dense",
            path_hash="dense",
            path=[],
            dense_score=0.99,
            score=0.40,
        )],
        hub_score=None,
    )

    result = strategy._prune_hubs(
        hubs=[dense_relevant, hybrid_relevant],
        alpha=3.0,
    )

    assert result == [hybrid_relevant]
    assert hybrid_relevant.hub_score == pytest.approx(0.95)
