"""
test_chroma_vector_store.py
"""
import pytest

from language_model import LLMProvider
from language_model.config.embedding_config import EmbeddingConfig
from knowledge_base.vector_store.storage.implementations.chroma_vector_store import ChromaVectorStore
from retrieval.implementations.HubLink.utils.hub_storage_manager import HubStorageManager
from tests.retrieval.implementations.HubLink.legacy.chroma_vector_store_old import ChromaVectorStoreOld

_VECTOR_STORE_NAME = "736960f8b73149a36889a4f2f5adc1d3_da9bf9ced3b86d4b6bfbcac1452859290775a394f73462884eb5f866fce6faed"
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
def store():
    return ChromaVectorStore(store_name=_VECTOR_STORE_NAME, distance_metric="cosine")


@pytest.fixture(scope="module")
def hub_storage_manager(store, embedding_model):
    return HubStorageManager(
        vector_store=store,
        embedding_model=embedding_model,
        diversity_penalty=_DIVERSITY_PENALTY,
    )


@pytest.fixture(scope="module")
def golden_master_store():
    return ChromaVectorStoreOld(
        store_name=_VECTOR_STORE_NAME,
        diversity_penalty=_DIVERSITY_PENALTY,
    )


@pytest.fixture(scope="module")
def query_embeddings(embedding_model):
    question = "Which papers evaluate the property security and have input data available?"
    components = ["Papers", "Property Security", "Input Data Available"]
    return embedding_model.embed_batch([question] + components)


def normalize_hub_path(hp):
    return {
        "path_hash": hp.path_hash,
        "path_text": hp.path_text,
        "score": round(hp.score, 6) if hp.score is not None else None,
        "embedded_text": hp.embedded_text,
    }


def normalize_list(paths):
    return [normalize_hub_path(p) for p in paths]


def normalize_clustered(clustered):
    return {hub_id: normalize_list(paths) for hub_id, paths in clustered.items()}


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

def test_similarity_search_hubs_with_excluded_hubs_identical(hub_storage_manager, golden_master_store, query_embeddings):
    excluded_hub_ids = ["R873379", "R872233", "R868446"]
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


def test_similarity_search_by_hub_entity_identical(hub_storage_manager, golden_master_store, query_embeddings):
    hub_entity_id = "R869225"
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

def test_similarity_search_by_hub_entity_with_excluded_path_identical(hub_storage_manager, golden_master_store, query_embeddings):
    hub_entity_id = "R872194"
    excluded_paths = ["6e6d6d77bb3060424fdb9a0596c2e4ca", "ebeb619a30cdb1a85e49589b140de47d", "251d12c15a5df63eac8057a031784f88", "f3227b988c474b99b06313109a03b04e", "da883087cb992537a48db504ccab0c0e", "11b81d281496e9abcb8b224a9d7f92a4", "74da87038db39367b1e8901e8e9e0c2d", "b349342d3d362d7f69550f70ef59fcad"]
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