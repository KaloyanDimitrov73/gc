"""
Unit tests for GraphExplorer.

These tests exercise the adjacency-index logic in isolation by replacing the
real HubLink graph object with lightweight fakes that expose the same
``cache_manager`` / ``cache_subgraph_key`` interface.  No network or database
access is required.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
HUBLINK_ROOT = PROJECT_ROOT / "backend" / "HublinkImplementation"
for candidate in (PROJECT_ROOT, HUBLINK_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from backend.app.modules.graph_explore.infrastructure.orkg.graph_explorer import GraphExplorer


class FakeCacheManager:
    """Minimal stand-in for the HubLink cache manager.

    Returns a fixed lookup table for the expected subgraph key and raises an
    assertion error for any other key, so tests fail loudly if the production
    code requests an unexpected cache entry.
    """

    def __init__(self, table):
        self._table = table

    def get_table(self, meta_key):
        assert meta_key == "fake_subgraph"
        return self._table


class FakeGraph:
    """Minimal stand-in for a HubLink graph object.

    Exposes the two attributes that ``GraphExplorer`` reads: the subgraph cache
    key and a ``FakeCacheManager`` backed by the caller-supplied table.
    """

    def __init__(self, table):
        self.cache_subgraph_key = "fake_subgraph"
        self.cache_manager = FakeCacheManager(table)


def test_graph_explorer_returns_neighbors_for_subject_and_object_matches():
    """GraphExplorer.get_neighbor_nodes returns all nodes incident to the queried node.

    Given two triples where R1 appears as subject in one and object in the
    other, the result must include R1 itself plus every node it is directly
    connected to (R2 and R3), with the correct directed connections attached.
    """
    graph = FakeGraph(
        {
            "triple-1": {
                "entity_subject": {"uid": "R1", "text": "Paper A", "knowledge_types": []},
                "predicate": "relatedTo",
                "entity_object": {"uid": "R2", "text": "Paper B", "knowledge_types": []},
            },
            "triple-2": {
                "entity_subject": {"uid": "R3", "text": "Paper C", "knowledge_types": []},
                "predicate": "cites",
                "entity_object": {"uid": "R1", "text": "Paper A", "knowledge_types": []},
            },
        }
    )

    explorer = GraphExplorer(graph)

    nodes = explorer.get_neighbor_nodes(node_id="R1")
    nodes_by_id = {node.id: node for node in nodes}

    assert set(nodes_by_id) == {"R1", "R2", "R3"}
    assert nodes_by_id["R1"].connections[0].target_id == "R2"
    assert nodes_by_id["R3"].connections[0].target_id == "R1"


def test_graph_explorer_returns_single_node_when_no_edges_exist():
    """GraphExplorer.get_neighbor_nodes returns a single isolated node when it has no edges.

    When the cache contains only a standalone entity (no triple payload) for
    the requested node id, the result must be a list with that one node and an
    empty connections list.
    """
    graph = FakeGraph(
        {
            "node-1": {"uid": "R9", "text": "Standalone", "knowledge_types": []},
        }
    )

    explorer = GraphExplorer(graph)

    nodes = explorer.get_neighbor_nodes(node_id="R9")

    assert len(nodes) == 1
    assert nodes[0].id == "R9"
    assert nodes[0].connections == []
