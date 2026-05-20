"""
Unit tests for GraphLoader.load_graph cache-resolution logic.

The tests cover the four mutually exclusive branches:
1. force_cache_update=True  → always re-download from ORKG, ignoring any existing cache.
2. SQLite cache present     → return the cached graph immediately without re-downloading.
3. No SQLite, JSON exists   → hydrate SQLite tables from the JSON file.
4. No cache at all          → download from ORKG (first-time setup).

External I/O (ORKGRemoteGraph, file system) is fully mocked so these tests run
offline without a live ORKG connection.
"""
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


pytestmark = pytest.mark.unit


# Ensure `sqa_system` imports resolve from backend/submodules/HublinkImplementation.
PROJECT_ROOT = Path(__file__).resolve().parents[6]
HUBLINK_IMPL_DIR = PROJECT_ROOT / "backend" / "submodules" / "HublinkImplementation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HUBLINK_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(HUBLINK_IMPL_DIR))

from backend.app.modules.graph_load.infrastructure.orkg.graph_loader import (
    GraphLoader,
)


def _make_config(force_cache_update: bool = False):
    """Helper: build a minimal config mock."""
    kg_config = MagicMock()
    kg_config.additional_params = {"force_cache_update": force_cache_update}
    kg_config.config_hash = "testhash"
    config = MagicMock()
    config.knowledge_graph_config = kg_config
    return config


def _make_graph_mock(has_sqlite_cache: bool = False):
    """Helper: build a minimal ORKGRemoteGraph mock."""
    graph = MagicMock()
    graph.cache_subgraph_key = "orkg_subgraph_testhash"
    graph.cache_publication_roots_key = "orkg_publications_root_cache_testhash"
    graph.paper_type = "Paper"
    if has_sqlite_cache:
        graph.cache_manager.get_table.return_value = {"key": "value"}
    else:
        graph.cache_manager.get_table.return_value = None
    return graph


def test_force_cache_update_downloads_from_orkg():
    """When force_cache_update=True, always re-download regardless of existing cache."""
    loader = GraphLoader()
    config = _make_config(force_cache_update=True)
    graph = _make_graph_mock(has_sqlite_cache=True)

    with patch(
        "backend.app.modules.graph_load.infrastructure.orkg.graph_loader.ORKGRemoteGraph",
        return_value=graph,
    ):
        result = loader.load_graph(config)

    graph.cache_subgraph.assert_called_once()
    assert result is graph


def test_uses_sqlite_cache_when_available():
    """When sqlite cache exists and force_cache_update=False, return immediately."""
    loader = GraphLoader()
    config = _make_config(force_cache_update=False)
    graph = _make_graph_mock(has_sqlite_cache=True)

    with patch(
        "backend.app.modules.graph_load.infrastructure.orkg.graph_loader.ORKGRemoteGraph",
        return_value=graph,
    ):
        result = loader.load_graph(config)

    graph.cache_subgraph.assert_not_called()
    assert result is graph


def test_hydrates_from_json_when_sqlite_empty_and_file_exists(tmp_path):
    """When no sqlite cache but JSON file exists, hydrate from JSON."""
    loader = GraphLoader()
    config = _make_config(force_cache_update=False)
    graph = _make_graph_mock(has_sqlite_cache=False)

    json_path = str(tmp_path / "testhash.json")
    Path(json_path).write_text("[]")

    with patch(
        "backend.app.modules.graph_load.infrastructure.orkg.graph_loader.ORKGRemoteGraph",
        return_value=graph,
    ):
        with patch.object(loader, "_get_orkg_cache_path", return_value=json_path):
            with patch.object(loader, "_hydrate_orkg_cache_tables_from_json") as hydrate_mock:
                result = loader.load_graph(config)

    hydrate_mock.assert_called_once_with(graph, json_path)
    graph.cache_subgraph.assert_not_called()
    assert result is graph


def test_downloads_from_orkg_when_no_cache_at_all(tmp_path):
    """When neither sqlite nor JSON cache exists, download from ORKG (first-time setup)."""
    loader = GraphLoader()
    config = _make_config(force_cache_update=False)
    graph = _make_graph_mock(has_sqlite_cache=False)

    non_existent_path = str(tmp_path / "missing.json")

    with patch(
        "backend.app.modules.graph_load.infrastructure.orkg.graph_loader.ORKGRemoteGraph",
        return_value=graph,
    ):
        with patch.object(loader, "_get_orkg_cache_path", return_value=non_existent_path):
            result = loader.load_graph(config)

    graph.cache_subgraph.assert_called_once()
    assert result is graph
