"""
test_hub_storage_legacy.py
"""
import pytest

from core.data.models import Knowledge, Triple
from hublink.core.models.entity_with_direction import EntityWithDirection
from hublink.core.utils.hub_path_util import path_to_hash
from language_model import LLMProvider
from language_model.config.embedding_config import EmbeddingConfig
from knowledge_base.vector_store.storage.implementations.chroma_vector_store import ChromaVectorStore
from hublink.core.hub_storage_manager import HubStorageManager
from tests.hublink.retrieval.legacy.chroma_vector_store_old import ChromaVectorStoreOld

_DIVERSITY_PENALTY = 0.05

_DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    name="openai_text-embedding-3-large",
    additional_params={},
    endpoint="OpenAI",
    name_model="text-embedding-3-large",
)

@pytest.fixture(scope="module")
def embedding_model():
    return LLMProvider().get_embeddings(embedding_config=_DEFAULT_EMBEDDING_CONFIG)


@pytest.fixture(scope="module")
def hub_a():
    entity = Knowledge(uid="TEST_R001", text="Paper A", knowledge_types=["Paper"])
    field = Knowledge(uid="TEST_F001", text="Property Security", knowledge_types=["ResearchField"])
    triple = Triple(entity_subject=entity, predicate="research field", entity_object=field)
    root = EntityWithDirection(entity=entity, left=True, path_from_topic=[triple])
    path_text = "Paper A is associated with the research field of Property Security."
    return root, triple, path_text


@pytest.fixture(scope="module")
def hub_b():
    entity = Knowledge(uid="TEST_R002", text="Paper B", knowledge_types=["Paper"])
    data = Knowledge(uid="TEST_D001", text="Input Data Available", knowledge_types=["Property"])
    triple = Triple(entity_subject=entity, predicate="has property", entity_object=data)
    root = EntityWithDirection(entity=entity, left=True, path_from_topic=[triple])
    path_text = "Paper B has input data available."
    return root, triple, path_text


@pytest.fixture(scope="module")
def hub_c():
    entity = Knowledge(uid="TEST_R003", text="Paper C", knowledge_types=["Paper"])
    field = Knowledge(uid="TEST_F002", text="Usability", knowledge_types=["ResearchField"])
    triple = Triple(entity_subject=entity, predicate="research field", entity_object=field)
    root = EntityWithDirection(entity=entity, left=True, path_from_topic=[triple])
    path_text = "Paper C is associated with the research field of Usability."
    return root, triple, path_text


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    store_path = tmp_path_factory.mktemp("chroma_new")
    return ChromaVectorStore(
        store_name="test_store_new",
        collection_name="test_collection_new",
        distance_metric="cosine",
        store_path=str(store_path),
    )

@pytest.fixture(scope="module")
def golden_master_store(tmp_path_factory, embedding_model):
    store_path = tmp_path_factory.mktemp("chroma_old")
    return ChromaVectorStoreOld(
        store_name="test_store_old",
        diversity_penalty=_DIVERSITY_PENALTY,
        store_path=str(store_path),
        embedding_model=embedding_model
    )


@pytest.fixture(scope="module")
def hub_storage_manager(store, embedding_model):
    return HubStorageManager(
        vector_store=store,
        embedding_config=_DEFAULT_EMBEDDING_CONFIG,
        diversity_penalty=_DIVERSITY_PENALTY,
    )

@pytest.fixture(scope="module", autouse=True)
def seed_stores(hub_storage_manager, golden_master_store, hub_a, hub_b, hub_c):
    for root, triple, path_text in (hub_a, hub_b, hub_c):
        hub_storage_manager.store_hub_batch(root, [[triple]], [path_text])
        golden_master_store.build_and_store_hubs([path_text], [[triple]], root)

@pytest.fixture(scope="module")
def query_embeddings(embedding_model):
    question = "Which papers evaluate the property security and have input data available?"
    components = ["Papers", "Property Security", "Input Data Available"]
    return embedding_model.embed_batch([question] + components)


def normalize_hub_path(hp):
    return {
        "path_hash": hp.path_hash,
        "path_text": hp.path_text,
        "score": round(hp.score, 3) if hp.score is not None else None,
        "embedded_text": hp.embedded_text,
    }


def normalize_list(paths):
    return [normalize_hub_path(p) for p in paths]


def normalize_clustered(clustered):
    return {hub_id: normalize_list(paths) for hub_id, paths in clustered.items()}


def test_stores_contain_identical_data_after_seeding(
    golden_master_store, hub_storage_manager, hub_a, hub_b, hub_c
):

    assert golden_master_store.collection.count() == hub_storage_manager.vector_store.count()

    for root, _, _ in (hub_a, hub_b, hub_c):
        hub_id = root.entity.uid

        old_paths, old_embeddings = golden_master_store.get_all_hub_paths_from_hub(hub_id)
        new_paths, new_embeddings = hub_storage_manager.get_all_hub_paths_from_hub(hub_id)

        assert len(old_paths) == len(new_paths)

        old_sorted = sorted(normalize_list(old_paths), key=lambda d: d["path_hash"])
        new_sorted = sorted(normalize_list(new_paths), key=lambda d: d["path_hash"])
        assert old_sorted == new_sorted

        assert len(old_embeddings) == len(new_embeddings)

def test_similarity_search_hubs_identical(hub_storage_manager, golden_master_store, query_embeddings):
    excluded_hub_ids = []
    n_results = 10

    old_result = golden_master_store.similarity_search_hubs(
        query_embeddings=query_embeddings,
        excluded_hub_ids=excluded_hub_ids,
        n_results=n_results,
    )
    new_result = hub_storage_manager.similarity_search_hubs(
        query_embeddings=query_embeddings,
        excluded_hub_ids=excluded_hub_ids,
        n_results=n_results,
    )

    assert normalize_clustered(old_result) == normalize_clustered(new_result)

def test_similarity_search_hubs_with_excluded_hubs_identical(hub_storage_manager, golden_master_store, query_embeddings, hub_a, hub_b):
    excluded_hub_ids = [hub_a[0].entity.uid, hub_b[0].entity.uid]
    n_results = 10

    old_result = golden_master_store.similarity_search_hubs(
        query_embeddings=query_embeddings,
        excluded_hub_ids=excluded_hub_ids,
        n_results=n_results,
    )
    new_result = hub_storage_manager.similarity_search_hubs(
        query_embeddings=query_embeddings,
        excluded_hub_ids=excluded_hub_ids,
        n_results=n_results,
    )

    assert normalize_clustered(old_result) == normalize_clustered(new_result)


def test_similarity_search_by_hub_entity_identical(hub_storage_manager, golden_master_store, query_embeddings, hub_a):
    hub_entity_id = hub_a[0].entity.uid
    n_results = 10

    old_result = golden_master_store.similarity_search_by_hub_entity(
        query_embeddings=query_embeddings,
        hub_entity_id=hub_entity_id,
        n_results=n_results,
    )
    new_result = hub_storage_manager.similarity_search_by_hub_entity(
        query_embeddings=query_embeddings,
        hub_entity_id=hub_entity_id,
        n_results=n_results,
    )

    assert normalize_list(old_result) == normalize_list(new_result)

def test_similarity_search_by_hub_entity_with_excluded_path_identical(hub_storage_manager, golden_master_store, query_embeddings, hub_b):
    root, triple, _ = hub_b
    hub_entity_id = root.entity.uid
    excluded_paths = [path_to_hash([triple])]
    n_results = 10

    old_result = golden_master_store.similarity_search_by_hub_entity(
        query_embeddings=query_embeddings,
        hub_entity_id=hub_entity_id,
        n_results=n_results,
        excluded_path_hashs=excluded_paths
    )
    new_result = hub_storage_manager.similarity_search_by_hub_entity(
        query_embeddings=query_embeddings,
        hub_entity_id=hub_entity_id,
        n_results=n_results,
        excluded_path_hashs=excluded_paths
    )

    assert normalize_list(old_result) == normalize_list(new_result)