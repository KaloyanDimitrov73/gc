"""
test_chroma_vector_store.py
"""
import pytest

from core.data.models import Knowledge, Triple
from hublink.core.models.entity_with_direction import EntityWithDirection
from hublink.core.models.hub_path import HubPath
from knowledge_base.vector_store.storage.implementations.chroma_vector_store import ChromaVectorStore
from language_model import LLMProvider
from language_model.config.embedding_config import EmbeddingConfig
from hublink.core.utils.hub_path_util import path_to_hash
from hublink.core.hub_storage_manager import HubStorageManager


_DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    name="openai_text-embedding-3-small",
    additional_params={},
    endpoint="OpenAI",
    name_model="text-embedding-3-small"
)

@pytest.fixture
def paper_entity():
    return Knowledge(
        uid="R868364",
        text="A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems",
        knowledge_types=["Paper"],
    )


@pytest.fixture
def research_field_entity():
    return Knowledge(
        uid="R659055",
        text="Software Architecture and Design",
        knowledge_types=["ResearchField"],
    )


@pytest.fixture
def venue_entity():
    return Knowledge(
        uid="R820814",
        text="International Conference on Software Architecture (ICSA)",
        knowledge_types=["Venue"],
    )

@pytest.fixture
def year_entity():
    return Knowledge(
        uid="L1519939",
        text="2020",
        knowledge_types=[],
    )


@pytest.fixture
def hub_root(paper_entity, research_field_path_triple):
    return EntityWithDirection(entity=paper_entity, left=True, path_from_topic=[research_field_path_triple])


@pytest.fixture
def research_field_path_triple(paper_entity, research_field_entity):
    return Triple(
            entity_subject=paper_entity,
            predicate="research field",
            entity_object=research_field_entity,
        )

@pytest.fixture
def venue_path_triple(paper_entity, venue_entity):
    return Triple(
            entity_subject=paper_entity,
            predicate="venue",
            entity_object=venue_entity,
        )

@pytest.fixture
def publication_year_path_triple(paper_entity, year_entity):
    return Triple(
            entity_subject=paper_entity,
            predicate="publication year",
            entity_object=year_entity,
        )


@pytest.fixture
def research_field_hub_path(research_field_path_triple):
    return HubPath(
        path_text='A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems is associated with the research field of Software Architecture and Design.',
        path_hash=path_to_hash([research_field_path_triple]),
        path=[research_field_path_triple],
        embedded_text=None,
        score=None
    )

@pytest.fixture
def venue_hub_path(venue_path_triple):
    return HubPath(
        path_text='The paper titled "A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems" was presented at the International Conference on Software Architecture (ICSA).',
        path_hash=path_to_hash([venue_path_triple]),
        path=[venue_path_triple],
        embedded_text=None,
        score=None
    )

@pytest.fixture
def publication_year_hub_path(publication_year_path_triple):
    return HubPath(
        path_text='The work titled "A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems" was published in 2020.',
        path_hash=path_to_hash([publication_year_path_triple]),
        path=[publication_year_path_triple],
        embedded_text=None,
        score=None
    )


@pytest.fixture
def second_paper_entity():
    return Knowledge(
        uid="R868251",
        text="A Framework for Tunable Anomaly Detection",
        knowledge_types=["Paper"],
    )


@pytest.fixture
def second_research_field_path_triple(second_paper_entity, research_field_entity):
    return Triple(
        entity_subject=second_paper_entity,
        predicate="research field",
        entity_object=research_field_entity,
    )


@pytest.fixture
def second_hub_root(second_paper_entity, second_research_field_path_triple):
    return EntityWithDirection(
        entity=second_paper_entity, left=True, path_from_topic=[second_research_field_path_triple]
    )


@pytest.fixture
def second_research_field_hub_path(second_research_field_path_triple):
    return HubPath(
        path_text='A Framework for Tunable Anomaly Detection is part of the research field of Software Architecture and Design.',
        path_hash=path_to_hash([second_research_field_path_triple]),
        path=[second_research_field_path_triple],
        embedded_text=None,
        score=None
    )


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(
        store_name="test_store",
        collection_name="test_collection",
        distance_metric="cosine",
        store_path=str(tmp_path / "chroma_test"),
    )

@pytest.fixture
def embedding_model(tmp_path):
    return LLMProvider().get_embeddings(embedding_config=_DEFAULT_EMBEDDING_CONFIG)

@pytest.fixture
def hub_storage_manager(store, embedding_model):
    return HubStorageManager(
        vector_store=store,
        embedding_model=embedding_model,
        diversity_penalty=0.1,
    )


def test_build_storage_record_form_path(
        hub_storage_manager, store, hub_root, research_field_path_triple, research_field_hub_path,
):
    texts, keys, metadatas = hub_storage_manager._build_storage_record_form_path(hub_root, [research_field_path_triple], research_field_hub_path.path_text)

    triple_text = texts[-1]
    triple_metadata = metadatas[-1]

    assert len(texts) == 5
    assert len(keys) == 5
    assert len(metadatas) == 5

    assert texts == [
        "A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems is associated with the research field of Software Architecture and Design.",
        "A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems",
        "Software Architecture and Design",
        "research field",
        "(A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems, research field, Software Architecture and Design)",
    ]

    assert keys[0] == path_to_hash([research_field_path_triple])

    assert metadatas[0]["path_hash"] == keys[0]
    assert metadatas[0]["path_text"] == research_field_hub_path.path_text
    assert metadatas[0]["hub_entity"] == hub_root.entity.uid
    assert metadatas[0]["length"] == 1
    assert metadatas[0]["embedded_text"] == research_field_hub_path.path_text

    assert triple_metadata["path_hash"] == metadatas[0]["path_hash"]
    assert triple_metadata["path_text"] == metadatas[0]["path_text"]


def test_deduplicate_entries_removes_duplicate_keys_keeps_first_occurrence(hub_storage_manager):
    texts = ["a", "b", "a-duplicate-content", "c"]
    keys = ["k1", "k2", "k1", "k3"]
    metadatas = [{"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}]

    dedup_texts, dedup_keys, dedup_metadatas = hub_storage_manager._deduplicate_entries(
        texts, keys, metadatas
    )

    assert dedup_keys == ["k1", "k2", "k3"]
    assert dedup_texts == ["a", "b", "c"]
    assert dedup_metadatas == [{"v": 1}, {"v": 2}, {"v": 4}]


def test_deduplicate_entries_with_no_duplicates_returns_unchanged(hub_storage_manager):
    texts = ["a", "b", "c"]
    keys = ["k1", "k2", "k3"]
    metadatas = [{"v": 1}, {"v": 2}, {"v": 3}]

    dedup_texts, dedup_keys, dedup_metadatas = hub_storage_manager._deduplicate_entries(
        texts, keys, metadatas
    )

    assert (dedup_texts, dedup_keys, dedup_metadatas) == (texts, keys, metadatas)

def test_store_hub_batch_single_path(
        hub_storage_manager, store, hub_root, research_field_path_triple, research_field_hub_path,
):
    assert store.count() == 0

    hub_storage_manager.store_hub_batch(hub_root, [[research_field_path_triple]], [research_field_hub_path.path_text])

    assert store.count() == 5


def test_store_hub_batch_multiple_paths_deduplicates_shared_entity(
        hub_storage_manager, store, hub_root,
        research_field_path_triple, venue_path_triple,
        research_field_hub_path, venue_hub_path,
        publication_year_path_triple, publication_year_hub_path,
):
    hub_storage_manager.store_hub_batch(
        hub_root,
        [[research_field_path_triple], [venue_path_triple], [publication_year_path_triple]],
        [research_field_hub_path.path_text, venue_hub_path.path_text, publication_year_hub_path.path_text],
    )

    # 3 full paths + 7 eindeutige Entities (Paper-Titel wird nur 1x gezählt) + 3 Triples = 13
    assert store.count() == 13


def test_store_hub_batch_with_empty_paths_does_nothing(hub_storage_manager, store, hub_root):
    hub_storage_manager.store_hub_batch(hub_root, [], [])

    assert store.count() == 0


def test_retrieve_one_hub_path_returns_matching_path(
        hub_storage_manager, hub_root, research_field_path_triple, research_field_hub_path,
):
    hub_storage_manager.store_hub_batch(hub_root, [[research_field_path_triple]], [research_field_hub_path.path_text])

    path_hash = path_to_hash([research_field_path_triple])
    result = hub_storage_manager.retrieve_one_hub_path(path_hash)

    assert result is not None
    assert result.path_hash == path_hash
    assert result.path_text == research_field_hub_path.path_text

def test_retrieve_one_hub_path_returns_none_for_unknown_hash(hub_storage_manager, store):
    assert hub_storage_manager.retrieve_one_hub_path("does-not-exist") is None


def test_retrieve_hub_paths_by_keys_batch_with_missing_key(
        hub_storage_manager, hub_root, research_field_path_triple, venue_path_triple,
        research_field_hub_path, venue_hub_path,
):
    hub_storage_manager.store_hub_batch(
        hub_root,
        [[research_field_path_triple], [venue_path_triple]],
        [research_field_hub_path.path_text, venue_hub_path.path_text],
    )

    key_research_field = path_to_hash([research_field_path_triple])
    key_venue = path_to_hash([venue_path_triple])
    missing_key = "not-in-store"

    results = hub_storage_manager.retrieve_hub_paths_by_keys([key_research_field, key_venue, missing_key])

    assert set(results.keys()) == {key_research_field, key_venue, missing_key}
    assert results[key_research_field].path_text == research_field_hub_path.path_text
    assert results[key_venue].path_text == venue_hub_path.path_text
    assert results[missing_key] is None


def test_get_all_hub_paths_from_hub_returns_empty_for_unknown_hub(hub_storage_manager, store):
    hub_paths, embeddings = hub_storage_manager.get_all_hub_paths_from_hub("unknown-hub-id")

    assert hub_paths == []
    assert embeddings == []


def test_get_all_hub_paths_from_hub_returns_records_and_embeddings(
        hub_storage_manager, hub_root, research_field_path_triple, research_field_hub_path,
):
    hub_storage_manager.store_hub_batch(hub_root, [[research_field_path_triple]], [research_field_hub_path.path_text])

    hub_paths, embeddings = hub_storage_manager.get_all_hub_paths_from_hub(hub_root.entity.uid)

    assert len(hub_paths) == 5
    assert len(embeddings) == 5
    assert all(hp.path_text == research_field_hub_path.path_text for hp in hub_paths)

def test_delete_data_from_hub_removes_only_matching_hub(
        hub_storage_manager, store, hub_root, second_hub_root,
        research_field_path_triple, research_field_hub_path,
        second_research_field_path_triple, second_research_field_hub_path,
):
    hub_storage_manager.store_hub_batch(hub_root, [[research_field_path_triple]], [research_field_hub_path.path_text])
    hub_storage_manager.store_hub_batch(
        second_hub_root, [[second_research_field_path_triple]], [second_research_field_hub_path.path_text]
    )
    assert store.count() == 10

    hub_storage_manager.delete_data_from_hub(hub_root.entity.uid)

    assert store.count() == 5
    deleted_hub = hub_storage_manager.get_all_hub_paths_from_hub(hub_root.entity.uid)[0]
    assert deleted_hub == []
    remaining_hub = hub_storage_manager.get_all_hub_paths_from_hub(second_hub_root.entity.uid)[0]
    assert len(remaining_hub) == 5


def test_similarity_search_by_hub_entity_finds_best_match(
        hub_storage_manager, hub_root, research_field_path_triple, venue_path_triple,
        research_field_hub_path, venue_hub_path,
):
    hub_storage_manager.store_hub_batch(
        hub_root,
        [[research_field_path_triple], [venue_path_triple]],
        [research_field_hub_path.path_text, venue_hub_path.path_text],
    )

    query_embedding = hub_storage_manager.embedding_model.embed_batch([research_field_hub_path.path_text])

    results = hub_storage_manager.similarity_search_by_hub_entity(
        query_embeddings=query_embedding,
        hub_entity_id=hub_root.entity.uid,
        n_results=3,
    )

    assert len(results) > 0

    assert results[0].path_text == research_field_hub_path.path_text
    assert results[0].score == pytest.approx(1.0, abs=0.05)


def test_similarity_search_by_hub_entity_respects_excluded_hashes(
        hub_storage_manager, hub_root, research_field_path_triple, venue_path_triple,
        research_field_hub_path, venue_hub_path,
):
    hub_storage_manager.store_hub_batch(
        hub_root,
        [[research_field_path_triple], [venue_path_triple]],
        [research_field_hub_path.path_text, venue_hub_path.path_text],
    )

    query_embedding = hub_storage_manager.embedding_model.embed_batch([research_field_hub_path.path_text])
    excluded_hash = path_to_hash([research_field_path_triple])

    results = hub_storage_manager.similarity_search_by_hub_entity(
        query_embeddings=query_embedding,
        hub_entity_id=hub_root.entity.uid,
        n_results=5,
        excluded_path_hashs=[excluded_hash],
    )

    assert all(hp.path_hash != excluded_hash for hp in results)


def test_similarity_search_hubs_excludes_given_hub_ids(
        hub_storage_manager, hub_root, second_hub_root,
        research_field_path_triple, second_research_field_path_triple,
        research_field_hub_path, second_research_field_hub_path,
):
    hub_storage_manager.store_hub_batch(hub_root, [[research_field_path_triple]], [research_field_hub_path.path_text])
    hub_storage_manager.store_hub_batch(
        second_hub_root, [[second_research_field_path_triple]], [second_research_field_hub_path.path_text]
    )

    query_embedding = hub_storage_manager.embedding_model.embed_batch(["Software Architecture and Design research"])

    results = hub_storage_manager.similarity_search_hubs(
        query_embeddings=query_embedding,
        excluded_hub_ids=[hub_root.entity.uid],
        n_results=5,
    )

    assert hub_root.entity.uid not in results
    assert second_hub_root.entity.uid in results

