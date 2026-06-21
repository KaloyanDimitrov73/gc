"""
Utilities for exploring the cached ORKG graph that HubLink already loads.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from core import Knowledge, Triple

from backend.app.contracts.schemas import GraphNode
from backend.app.modules.graph_explore.infrastructure.orkg.node_builder import (
    build_graph_node,
    build_graph_nodes_from_edges,
)


class GraphExplorer:
    """Builds a lightweight adjacency index over the cached graph on demand."""

    def __init__(self, graph: Any):
        """
        Args:
            graph: The HubLink graph object exposing a ``cache_manager`` and
                   ``cache_subgraph_key`` used to populate the adjacency index.
        """
        self._graph = graph
        self._index_lock = threading.RLock()
        self._is_index_ready = False
        self._entities_by_id: Dict[str, Knowledge] = {}
        self._incident_triples_by_node_id: Dict[str, List[Triple]] = defaultdict(list)

    def get_neighbor_nodes(self, *, node_id: str) -> List[GraphNode]:
        """
        Return the direct-neighbor graph fragment centered around the given node.

        The response is intentionally shaped like the existing graph payload so the
        frontend can merge it into the already rendered answer graph.
        """
        self._ensure_index()
        collected_triples: Dict[Tuple[str, str, str], Triple] = {}

        for triple in self._incident_triples_by_node_id.get(node_id, []):
            triple_key = (
                triple.entity_subject.uid,
                triple.predicate,
                triple.entity_object.uid,
            )
            collected_triples[triple_key] = triple

        if collected_triples:
            return build_graph_nodes_from_edges(
                [
                    (
                        self._knowledge_to_entity_dict(triple.entity_subject),
                        triple.predicate,
                        self._knowledge_to_entity_dict(triple.entity_object),
                        None,
                    )
                    for triple in collected_triples.values()
                ]
            )

        entity = self._entities_by_id.get(node_id)
        if entity is None:
            return []

        return [build_graph_node(self._knowledge_to_entity_dict(entity))]

    def _ensure_index(self) -> None:
        """Build the in-memory adjacency index from the graph cache if not already done.

        Iterates over every entry in the cache table, classifies each payload as either
        a ``Triple`` or a standalone ``Knowledge`` entity, and populates
        ``_entities_by_id`` and ``_incident_triples_by_node_id``. Idempotent — the index
        is built at most once per instance (guarded by ``_index_lock``).
        """
        with self._index_lock:
            if self._is_index_ready:
                return

            cache_table = self._graph.cache_manager.get_table(self._graph.cache_subgraph_key) or {}
            for raw_item in cache_table.values():
                payload = self._normalize_cached_item(raw_item)
                if payload is None:
                    continue

                if self._is_triple_payload(payload):
                    triple = Triple.model_validate(payload)
                    self._register_entity(triple.entity_subject)
                    self._register_entity(triple.entity_object)
                    self._incident_triples_by_node_id[triple.entity_subject.uid].append(triple)
                    self._incident_triples_by_node_id[triple.entity_object.uid].append(triple)
                    continue

                if self._is_knowledge_payload(payload):
                    self._register_entity(Knowledge.model_validate(payload))

            self._is_index_ready = True

    def _register_entity(self, entity: Knowledge) -> None:
        """Insert *entity* into the id-lookup map, merging into an existing entry if present.

        When the same uid appears more than once (e.g. as both a triple subject and a
        standalone entity), the ``text`` and ``knowledge_types`` fields are merged so
        that the richest available metadata is retained.

        Args:
            entity: The ``Knowledge`` instance to register.
        """
        existing = self._entities_by_id.get(entity.uid)
        if existing is None:
            self._entities_by_id[entity.uid] = entity
            return

        if not existing.text and entity.text:
            existing.text = entity.text

        merged_types = set(existing.knowledge_types or [])
        merged_types.update(entity.knowledge_types or [])
        existing.knowledge_types = sorted(merged_types)

    @staticmethod
    def _normalize_cached_item(raw_item: Any) -> Optional[Dict[str, Any]]:
        """Coerce a raw cache entry to a plain ``dict``, or return ``None`` if not possible.

        Accepts entries that are already a ``dict`` or a JSON-encoded string.

        Args:
            raw_item: The value retrieved from the cache table.

        Returns:
            A ``dict`` representation of the item, or ``None`` if the item cannot be
            interpreted as a mapping.
        """
        if isinstance(raw_item, dict):
            return raw_item

        if isinstance(raw_item, str):
            try:
                decoded = json.loads(raw_item)
            except json.JSONDecodeError:
                return None
            return decoded if isinstance(decoded, dict) else None

        return None

    @staticmethod
    def _is_triple_payload(payload: Dict[str, Any]) -> bool:
        return (
            "predicate" in payload
            and "entity_subject" in payload
            and "entity_object" in payload
        )

    @staticmethod
    def _is_knowledge_payload(payload: Dict[str, Any]) -> bool:
        return "uid" in payload and "predicate" not in payload

    @staticmethod
    def _knowledge_to_entity_dict(entity: Knowledge) -> Dict[str, str]:
        return {
            "id": entity.uid,
            "label": entity.text or entity.uid,
            "type": "literal" if entity.uid.startswith("L") else "resource",
        }
