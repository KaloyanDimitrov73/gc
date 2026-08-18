import gc
import logging
import shutil
from pathlib import Path

import pytest

from backend.app.config.hublink.config_loader import HublinkConfigLoader
from backend.app.modules.graph_load.application.graph_load_service import GraphLoadService
from backend.app.modules.indexing.infrastructure.hublink.hub_indexing_service import HubIndexingService
from backend.app.modules.indexing.infrastructure.hublink.hub_store_service import HubStoreService


CONFIG_PATH = Path(__file__).parent / "data" / "test_config.json"
GRAPH_PATH = Path(__file__).parent / "data" / "test_graph.json"

CACHE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "cache" / "hublink_retriever" / "624ce22a30fc1e92ccc9b82ba737384d_da9bf9ced3b86d4b6bfbcac1452859290775a394f73462884eb5f866fce6faed"


@pytest.fixture(autouse=True)
def propagate_qasystem_logs(monkeypatch):
    """
    The custom QASystem logger (see core.logging.logging) sets
    propagate=False so its records don't leak to the root logger in
    production. That also means caplog - which listens on the root
    logger - never sees them. Temporarily re-enable propagation for
    the duration of each test so caplog works as expected.
    """
    monkeypatch.setattr(logging.getLogger("QASystem"), "propagate", True)

@pytest.fixture
def kg_retrieval_config():
    """
    Loads the test_config shared by all fixtures below.
    """

    return HublinkConfigLoader(config_file_path=CONFIG_PATH).load_kg_retrieval_config()


@pytest.fixture
def orkg_graph_load_service(kg_retrieval_config):
    """
    Loads the test_config and overrides the
    knowledge_graph_config so it's hydrated from local test JSON data.
    """

    return GraphLoadService(config=kg_retrieval_config)

@pytest.fixture
def vector_store_service(kg_retrieval_config):
    """
    Creates a fresh HubStoreService per test.

    Teardown closes the underlying Chroma client and deletes the on-disk
    cache so each test starts from an empty store. Runs after every test
    (function scope) - see chat notes if you want this shared across
    the module instead.
    """

    service = HubStoreService(config=kg_retrieval_config)

    yield service

    service.hub_storage_manager.vector_store.close()

    if CACHE_PATH.exists():
        try:
            shutil.rmtree(CACHE_PATH)
            print(f"DELETED: {CACHE_PATH}")
        except Exception as e:
            print(f"FAILED TO DELETE {CACHE_PATH}: {e}")
            raise
    else:
        print(f"CACHE_PATH DOES NOT EXIST: {CACHE_PATH}")

@pytest.fixture
def hub_indexing_service(orkg_graph_load_service, vector_store_service, kg_retrieval_config):
    """
    Loads the test_config and initialize an TestVectorStore.
    """

    return HubIndexingService(
            config=kg_retrieval_config,
            graph=orkg_graph_load_service.graph,
            hub_storage_manager=vector_store_service.hub_storage_manager,
        )


PAPER_ID_1 = "R873379"  # "Predicting the Performance of Privacy-Preserving..."

class TestHubIndexingServiceInit:
    def test_initial_indexing_when_vector_store_is_empty(
        self, orkg_graph_load_service, vector_store_service, kg_retrieval_config
    ):
        assert vector_store_service.hub_storage_manager.vector_store_is_empty()

        HubIndexingService(
            config=kg_retrieval_config,
            graph=orkg_graph_load_service.graph,
            hub_storage_manager=vector_store_service.hub_storage_manager,
        )

        assert vector_store_service.hub_storage_manager.vector_store.count() == 246

    def test_indexing_run_without_force_update_reuses_cache(self, hub_indexing_service, orkg_graph_load_service, caplog):
        graph = orkg_graph_load_service.graph

        # First run already happened during fixture construction (store was empty).
        # A second run without force_update should hit the cache and not error.
        with caplog.at_level(logging.DEBUG):
            indexed_count = hub_indexing_service.run_indexing(graph=graph, force_update=False)

        assert indexed_count > 0
        assert hub_indexing_service.hub_storage_manager.vector_store.count() == 246

    def test_reindexing_of_deleted_values_without_force_update_reuses_cache(self, vector_store_service, hub_indexing_service, orkg_graph_load_service, caplog):
        graph = orkg_graph_load_service.graph

        # Simulate a stale/missing hub: delete one hub's data from the vector store,
        # then verify that a second indexing run (without force_update) detects this
        # and rebuilds it from the graph, without erroring.

        vector_store_service.hub_storage_manager.delete_data_from_hub(hub_entity_id=PAPER_ID_1)

        assert hub_indexing_service.hub_storage_manager.vector_store.count() == 142

        with caplog.at_level(logging.DEBUG):
            indexed_count = hub_indexing_service.run_indexing(graph=graph, force_update=False)

        assert indexed_count > 0
        assert hub_indexing_service.hub_storage_manager.vector_store.count() == 246
