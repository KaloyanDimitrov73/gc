"""
test_chroma_vector_store.py
"""
import numpy as np
import pytest

from knowledge_base.vector_store.storage.implementations.chroma_vector_store import ChromaVectorStore
from knowledge_base.vector_store.storage.utils.filters import FilterCondition, FilterOperator, FilterGroup, \
    LogicalOperator


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(
        store_name="test_store",
        collection_name="test_collection",
        distance_metric="cosine",
        store_path=str(tmp_path / "chroma_test"),
    )


def test_initializes_client_and_collection(store):
    assert store.client is not None
    assert store.collection is not None
    assert store.count() == 0

def test_get_by_id_returns_none_if_not_found(store):
    result = store.get_records_with_metadata_by_ids(ids=["nonexistent"])
    assert result is None

def test_store_data_batch_empty_list(store):
    store.store_data_batch(ids=[], embeddings=[], metadatas=[])
    assert store.count() == 0

# Chroma stores embeddings as float32, not float64.
# As a result, minor differences in precision may occur (e.g., 0.4 → 0.40000001).
# Use np.allclose()
def test_store_data_get_by_id(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3])
    assert store.count() == 1

    result = store.get_records_with_metadata_by_ids(ids=["paper_1"])

    assert result is not None
    assert np.allclose(result.embeddings, [[0.1, 0.2, 0.3]])
    assert result.metadata == [None]

def test_store_data_overwrites_existing_id(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"topic": "Software Architecture and Design"})
    store.store_data("paper_1", [0.4, 0.5, 0.6], {"topic": "Software Analysis"})

    assert store.count() == 1

    result = store.get_records_with_metadata_by_ids(ids=["paper_1"])

    assert result is not None
    assert np.allclose(result.embeddings, [[0.4, 0.5, 0.6]])
    assert result.metadata == [{"topic": "Software Analysis"}]

def test_store_data_batch_and_get_by_ids(store):
    store.store_data_batch(
        ids=["paper_1", "paper_2", "paper_3"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.0, 0.6]],
        metadatas=[
            {"topic": "Software Architecture and Design"},
            {"topic": "Software Analysis"},
            None
        ]
    )

    assert store.count() == 3

    result = store.get_records_with_metadata_by_ids(ids=["paper_1", "paper_2", "paper_3"])

    assert result is not None
    assert result.ids == ["paper_1", "paper_2", "paper_3"]
    assert np.allclose(result.embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.0, 0.6]])
    assert result.metadata == [{"topic": "Software Architecture and Design"}, {"topic": "Software Analysis"}, None]

def test_get_records_with_metadata_by_filter(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"topic": "Software Architecture and Design"})
    store.store_data("paper_2", [0.4, 0.5, 0.6], {"topic": "Software Analysis"})
    store.store_data("paper_3", [0.1, 0.3, 0.5], {"topic": "Software Architecture and Design"})

    assert store.count() == 3

    where_filter = FilterCondition(
        field="topic",
        operator=FilterOperator.EQUALS,
        value="Software Architecture and Design",
    )

    result = store.get_records_with_metadata_by_filter(where_filter, limit=10)

    assert result is not None
    assert result.ids == ["paper_1", "paper_3"]
    assert np.allclose(result.embeddings, [[0.1, 0.2, 0.3], [0.1, 0.3, 0.5]])

def test_get_records_with_metadata_by_filter_no_match_returns_none(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"topic": "Software Architecture and Design"})

    where_filter = FilterCondition(
        field="category",
        operator=FilterOperator.EQUALS,
        value="nonexistent",
    )

    result = store.get_records_with_metadata_by_filter(where_filter)

    assert result is None


def test_delete_records_with_filter(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"topic": "Software Architecture and Design"})
    store.store_data("paper_2", [0.4, 0.5, 0.6], {"topic": "Software Analysis"})

    assert store.count() == 2

    where_filter = FilterCondition(
        field="topic",
        operator=FilterOperator.EQUALS,
        value="Software Architecture and Design",
    )

    store.delete_records_with_filter(where_filter)

    assert store.count() == 1
    assert store.get_records_with_metadata_by_ids(ids=["paper_1"]) is None
    assert store.get_records_with_metadata_by_ids(ids=["paper_2"]) is not None


def test_vector_similarity_search_returns_closest_first(store):
    store.store_data("far", [0.0, 1.0])
    store.store_data("close", [1.0, 0.0])

    assert store.count() == 2

    results = store.vector_similarity_search(
        query_embeddings=[[1.0, 0.0]],
        n_results=2,
    )

    assert len(results) == 1
    assert results[0].ids[0] == "close"
    assert results[0].ids[1] == "far"
    assert np.allclose(results[0].embeddings, [[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(results[0].distances, [0.0, 1.0])

def test_vector_similarity_search_respects_filter(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"topic": "Software Architecture and Design"})
    store.store_data("paper_2", [0.1, 0.2, 0.3], {"topic": "Software Analysis"})

    where_filter = FilterCondition(
        field="topic",
        operator=FilterOperator.EQUALS,
        value="Software Analysis",
    )

    results = store.vector_similarity_search(
        query_embeddings=[[0.1, 0.2, 0.3]],
        where_filter=where_filter,
        n_results=10,
    )

    assert len(results) == 1
    assert results[0].ids == ["paper_2"]


def test_vector_similarity_search_multiple_queries(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3])
    store.store_data("paper_2", [0.1, 0.0, 0.3])

    results = store.vector_similarity_search(
        query_embeddings=[[0.1, 0.2, 0.3], [0.1, 0.0, 0.3]],
        n_results=1,
    )

    assert len(results) == 2
    assert results[0].ids == ["paper_1"]
    assert results[1].ids == ["paper_2"]

def test_calculate_max_similarities_basic(store):
    embeddings_array = np.array([
        [1.0, 0.0],
        [0.4, 0.6],
        [0.0, 1.0],
    ])

    query_embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    result = store._calculate_max_similarities(
        query_embeddings=query_embeddings,
        embeddings_array=embeddings_array,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(query_embeddings), len(embeddings_array))

def test_manually_calculation_with_multiple_queries(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3])
    store.store_data("paper_2", [0.1, 0.0, 0.3])

    results = store.vector_similarity_search(
        query_embeddings=[[0.1, 0.2, 0.3], [0.1, 0.0, 0.3]],
        n_results=2,
    )

    assert len(results) == 2
    assert results[0].ids == ["paper_1", "paper_2"]
    assert results[1].ids == ["paper_2", "paper_1"]
    assert np.allclose(results[0].distances, [-1.7738799762412327e-08, 0.15484572488212722])
    assert np.allclose(results[1].distances, [-4.397469721695302e-08, 0.15484570270874687])

    manual_results = store._manually_calculate_similarity_score(
        query_embeddings=[[0.1, 0.2, 0.3], [0.1, 0.0, 0.3]],
        n_results=2,
    )

    assert manual_results[0].ids == results[0].ids
    assert manual_results[1].ids == results[1].ids

    np.allclose(manual_results[0].distances, results[0].distances)
    np.allclose(manual_results[1].distances, results[1].distances)

def test_not_equals_none_matches_everything(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"path_hash": "hash_123"})
    store.store_data("paper_2", [0.4, 0.5, 0.6], {"path_hash": "hash_456"})

    where_filter = FilterCondition(field="path_hash", operator=FilterOperator.NOT_EQUALS, value=None)

    result = store.get_records_with_metadata_by_filter(where_filter, limit=10)
    assert set(result.ids) == {"paper_1", "paper_2"}


def test_not_in_empty_list_matches_everything(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"path_hash": "hash_123"})
    store.store_data("paper_2", [0.4, 0.5, 0.6], {"path_hash": "hash_456"})

    where_filter = FilterCondition(field="path_hash", operator=FilterOperator.IS_NOT_IN_LIST, value=[])

    result = store.get_records_with_metadata_by_filter(where_filter, limit=10)
    assert set(result.ids) == {"paper_1", "paper_2"}


def test_equals_none_raises(store):
    where_filter = FilterCondition(field="path_hash", operator=FilterOperator.EQUALS, value=None)

    with pytest.raises(ValueError, match="Invalid filter"):
        store.get_records_with_metadata_by_filter(where_filter)


def test_in_empty_list_raises(store):
    where_filter = FilterCondition(field="path_hash", operator=FilterOperator.IS_IN_LIST, value=[])

    with pytest.raises(ValueError, match="Invalid filter"):
        store.get_records_with_metadata_by_filter(where_filter)


def test_greater_than_none_raises(store):
    where_filter = FilterCondition(field="score", operator=FilterOperator.GREATER_THAN, value=None)

    with pytest.raises(ValueError, match="Invalid filter"):
        store.get_records_with_metadata_by_filter(where_filter)


def test_delete_records_with_complex_nested_filter(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"topic": "Software Architecture", "year": 2021})
    store.store_data("paper_2", [0.4, 0.5, 0.6], {"topic": "Software Analysis", "year": 2019})
    store.store_data("paper_3", [0.7, 0.8, 0.9], {"topic": "Software Analysis", "year": 2022})
    store.store_data("paper_4", [1.0, 1.1, 1.2], {"topic": "Machine Learning", "year": 2024})


    complex_filter = FilterGroup(
        operator=LogicalOperator.AND,
        conditions=(
            FilterGroup(
                operator=LogicalOperator.OR,
                conditions=(
                    FilterCondition(
                        field="topic",
                        operator=FilterOperator.EQUALS,
                        value="Software Architecture",
                    ),
                    FilterCondition(
                        field="topic",
                        operator=FilterOperator.EQUALS,
                        value="Software Analysis",
                    ),
                ),
            ),
            FilterCondition(
                field="year",
                operator=FilterOperator.GREATER_THAN_OR_EQUAL,
                value=2021,
            ),
        ),
    )

    store.delete_records_with_filter(complex_filter)

    assert store.count() == 2

    assert store.get_records_with_metadata_by_ids(ids=["paper_1"]) is None
    assert store.get_records_with_metadata_by_ids(ids=["paper_3"]) is None

    assert store.get_records_with_metadata_by_ids(ids=["paper_2"]) is not None
    assert store.get_records_with_metadata_by_ids(ids=["paper_4"]) is not None


def test_get_records_with_and_not_in_filter(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"path_hash": "hash_123", "hub_entity": "hub_a"})
    store.store_data("paper_2", [0.4, 0.5, 0.6], {"path_hash": "hash_456", "hub_entity": "hub_a"})
    store.store_data("paper_3", [0.7, 0.8, 0.9], {"path_hash": "hash_257", "hub_entity": "hub_a"})
    store.store_data("paper_4", [1.0, 1.1, 1.2], {"path_hash": "hash_534", "hub_entity": "hub_b"})

    excluded_path_hashes = ["hash_123", "hash_257"]
    hub_entity_id = "hub_a"

    where_filter = FilterGroup(
        operator=LogicalOperator.AND,
        conditions=(
            FilterCondition(
                field="path_hash",
                operator=FilterOperator.IS_NOT_IN_LIST,
                value=excluded_path_hashes,
            ),
            FilterCondition(
                field="hub_entity",
                operator=FilterOperator.EQUALS,
                value=hub_entity_id,
            ),
        ),
    )

    result = store.get_records_with_metadata_by_filter(where_filter)

    assert result is not None
    assert result.ids == ["paper_2"]


def test_get_records_with_and_not_in_filter_empty(store):
    store.store_data("paper_1", [0.1, 0.2, 0.3], {"path_hash": "hash_123", "hub_entity": "hub_a"})
    store.store_data("paper_2", [0.4, 0.5, 0.6], {"path_hash": "hash_456", "hub_entity": "hub_a"})
    store.store_data("paper_3", [0.7, 0.8, 0.9], {"path_hash": "hash_257", "hub_entity": "hub_b"})
    store.store_data("paper_4", [1.0, 1.1, 1.2], {"path_hash": "hash_534", "hub_entity": "hub_a"})

    hub_entity_id = "hub_a"

    where_filter = FilterGroup(
        operator=LogicalOperator.AND,
        conditions=(
            FilterCondition(
                field="path_hash",
                operator=FilterOperator.IS_NOT_IN_LIST,
                value=[],
            ),
            FilterCondition(
                field="hub_entity",
                operator=FilterOperator.EQUALS,
                value=hub_entity_id,
            ),
        ),
    )

    result = store.get_records_with_metadata_by_filter(where_filter, limit=10)

    assert result is not None
    assert result.ids == ["paper_1", "paper_2", "paper_4"]