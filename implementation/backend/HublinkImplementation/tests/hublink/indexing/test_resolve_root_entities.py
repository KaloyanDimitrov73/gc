import logging
from pathlib import Path

import pytest

from backend.app.config.hublink.config_loader import HublinkConfigLoader
from backend.app.modules.graph_load.infrastructure.orkg.graph_loader import GraphLoader

from hublink.indexing.util.root_entity_util import resolve_root_entities
from knowledge_base.knowledge_graph.storage.implementations.orkg_remote_graph import (
    ORKGRemoteGraph,
)

CONFIG_PATH = Path(__file__).parent / "data" / "test_config.json"
GRAPH_PATH = Path(__file__).parent / "data" / "test_graph.json"


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
def orkg_graph():
    """
    Loads the test_config and overrides the
    knowledge_graph_config so it's hydrated from local test JSON data.
    """
    kg_retrieval_config = HublinkConfigLoader(
        config_file_path=CONFIG_PATH
    ).load_kg_retrieval_config()

    kg_config = kg_retrieval_config.knowledge_graph_config

    graph = ORKGRemoteGraph(kg_config)

    GraphLoader()._hydrate_orkg_cache_tables_from_json(graph, str(GRAPH_PATH))

    return graph


PAPER_ID_1 = "R873379"  # "Predicting the Performance of Privacy-Preserving..."
PAPER_ID_2 = "R873599"  # "Improving the Consistency and Usefulness..."
RESEARCH_FIELD_ID = "R659055"  # "Software Architecture and Design"
NON_EXISTENT_ID = "R000000"


class TestResolveRootEntitiesByIds:
    def test_resolves_single_valid_id(self, orkg_graph):
        result = resolve_root_entities(orkg_graph, root_entity_ids=[PAPER_ID_1])

        assert len(result) == 1
        assert result[0].uid == PAPER_ID_1

    def test_resolves_multiple_valid_ids(self, orkg_graph):
        result = resolve_root_entities(
            orkg_graph, root_entity_ids=[PAPER_ID_1, PAPER_ID_2]
        )

        resolved_ids = {entity.uid for entity in result}
        assert resolved_ids == {PAPER_ID_1, PAPER_ID_2}

    def test_skips_and_warns_on_unknown_id(self, orkg_graph, caplog):
        with caplog.at_level("WARNING", logger="QASystem"):
            result = resolve_root_entities(
                orkg_graph, root_entity_ids=[PAPER_ID_1, NON_EXISTENT_ID]
            )

        # Only the valid id is resolved
        assert len(result) == 1
        assert result[0].uid == PAPER_ID_1
        # A warning was logged for the missing one
        assert any(NON_EXISTENT_ID in record.getMessage() for record in caplog.records)

    def test_returns_empty_list_when_no_id_matches(self, orkg_graph, caplog):
        with caplog.at_level("WARNING", logger="QASystem"):
            result = resolve_root_entities(orkg_graph, root_entity_ids=[NON_EXISTENT_ID])

        assert result == []
        assert any("No root entities found" in record.getMessage() for record in caplog.records)


class TestResolveRootEntitiesByTypes:
    def test_resolves_entities_by_type(self, orkg_graph):
        result = resolve_root_entities(orkg_graph, root_entity_types=["Paper"])

        resolved_ids = {entity.uid for entity in result}
        # Both papers in the test graph should be found
        assert PAPER_ID_1 in resolved_ids
        assert PAPER_ID_2 in resolved_ids

    def test_resolves_entities_by_research_field_type(self, orkg_graph):
        result = resolve_root_entities(orkg_graph, root_entity_types=["ResearchField"])

        resolved_ids = {entity.uid for entity in result}
        assert RESEARCH_FIELD_ID in resolved_ids

    def test_returns_empty_list_for_unknown_type(self, orkg_graph, caplog):
        with caplog.at_level("WARNING", logger="QASystem"):
            result = resolve_root_entities(orkg_graph, root_entity_types=["NoSuchType"])

        assert result == []
        assert any("No root entities found" in record.getMessage() for record in caplog.records)


class TestResolveRootEntitiesCombined:
    def test_union_of_ids_and_types(self, orkg_graph):
        result = resolve_root_entities(
            orkg_graph,
            root_entity_ids=[PAPER_ID_1],
            root_entity_types=["ResearchField"],
        )

        resolved_ids = {entity.uid for entity in result}
        assert PAPER_ID_1 in resolved_ids
        assert RESEARCH_FIELD_ID in resolved_ids

    def test_raises_when_neither_ids_nor_types_given(self, orkg_graph):
        with pytest.raises(ValueError, match="Either root_entity_types or root_entity_ids"):
            resolve_root_entities(orkg_graph)