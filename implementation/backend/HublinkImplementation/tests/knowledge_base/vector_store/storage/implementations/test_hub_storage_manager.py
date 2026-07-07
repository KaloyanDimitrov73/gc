"""
test_chroma_vector_store.py
"""
import numpy as np
import pytest

from core.data.models import Knowledge, Triple
from knowledge_base.vector_store.storage.implementations.chroma_vector_store import ChromaVectorStore
from language_model import LLMProvider
from language_model.config.embedding_config import EmbeddingConfig
from retrieval.implementations.HubLink.models import EntityWithDirection, HubPath
from retrieval.implementations.HubLink.utils.hub_path_util import path_to_hash
from retrieval.implementations.HubLink.utils.hub_storage_manager import HubStorageManager


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
        path_hash='5c8cc6738ab64916d5cda4d0098623ee',
        path=[research_field_path_triple],
        embedded_text=None,
        score=None
    )

@pytest.fixture
def venue_hub_path(venue_path_triple):
    return HubPath(
        path_text='The paper titled "A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems" was presented at the International Conference on Software Architecture (ICSA).',
        path_hash='12ebf7a0cf37a5069c8ad164c8ec0879',
        path=[venue_path_triple],
        embedded_text=None,
        score=None
    )

@pytest.fixture
def publication_year_hub_path(publication_year_path_triple):
    return HubPath(
        path_text='The work titled "A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems" was published in 2020.',
        path_hash='9f481fa6ce19df43f55b619418429c59',
        path=[publication_year_path_triple],
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


def test_store_hub_batch_single_path_metadata_length(
        hub_storage_manager, store, hub_root, research_field_path_triple, research_field_hub_path,
):
    assert store.count() == 0

    hub_storage_manager.store_hub_batch(hub_root, [[research_field_path_triple]], [research_field_hub_path.path_text])

    assert store.count() == 5
