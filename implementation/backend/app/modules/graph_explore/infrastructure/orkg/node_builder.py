"""
Utility module for converting HubLink retrieval answers into
graph nodes and sources for the frontend visualization.
"""
from typing import Iterable, List, Tuple, Dict, Optional, Any
import re

import logging

from backend.app.contracts.schemas import GraphNode, NodeConnection

logger = logging.getLogger(__name__)
ENTITY_ID_PATTERN = re.compile(r"^[RL]\d+$")


def build_answer_nodes_sources(retrieval_answer) -> Tuple[str, List[GraphNode], List[str]]:
    """
    Convert a RetrievalAnswer into the (answer, nodes, sources) tuple
    expected by the query endpoint.

    Args:
        retrieval_answer: RetrievalAnswer returned by the HubLink retriever.

    Returns:
        Tuple of (answer_text, graph_nodes, source_identifiers).
        Returns a fallback tuple when the answer is empty/invalid.
    """
    if retrieval_answer.retriever_answer is None or not retrieval_answer.contexts:
        logger.info("SciGraphChat retriever did not return a valid answer")
        return (
            "SciGraphChat is not able to retrieve a valid answer.",
            [],
            [],
        )

    answer = retrieval_answer.retriever_answer
    nodes = _contexts_to_graph_nodes(retrieval_answer.contexts)
    # Prefer sources tracked by the answer generator (cited hubs only);
    # fall back to metadata-based extraction for backwards compatibility.
    sources = getattr(retrieval_answer, "sources", None) or _extract_sources(retrieval_answer.contexts)

    return answer, nodes, sources


def build_graph_node(entity: Dict[str, str], score: Optional[float] = None) -> GraphNode:
    """Build a single graph node without any connections."""
    return GraphNode(
        id=entity["id"],
        label=entity["label"],
        type=entity["type"],
        connections=[],
        score=score,
    )


def build_graph_nodes_from_edges(
    edges: Iterable[Tuple[Dict[str, str], str, Dict[str, str], Optional[float]]]
) -> List[GraphNode]:
    """
    Build graph nodes from normalized edge tuples.

    Each edge is represented as (subject_entity, predicate, object_entity, score),
    where subject/object entities are dictionaries with id/label/type keys.
    """
    nodes_by_id: Dict[str, GraphNode] = {}

    try:
        for subject, predicate, obj, score in edges:
            source_node = _get_or_create_entity_node(
                nodes_by_id=nodes_by_id,
                entity=subject,
                score=score,
            )
            _get_or_create_entity_node(
                nodes_by_id=nodes_by_id,
                entity=obj,
                score=score,
            )

            if source_node.type != "literal":
                connection = NodeConnection(
                    target_id=obj["id"],
                    relation=predicate,
                )
                if connection not in source_node.connections:
                    source_node.connections.append(connection)

    except Exception as e:
        logger.error(f"Error converting edges to nodes: {e}", exc_info=True)

    return list(nodes_by_id.values())




def _contexts_to_graph_nodes(contexts) -> List[GraphNode]:
    """
    Convert HubLink contexts to graph nodes for visualization.

    Only knowledge-graph triples are considered. Each RDF-like triple is
    converted into subject/object nodes plus a predicate edge. Entity IDs are
    deduplicated across all contexts, and node depth is computed from the
    resulting graph topology.
    """
    parsed_edges: List[Tuple[Dict[str, str], str, Dict[str, str], Optional[float]]] = []

    try:
        for context in contexts:
            if not _is_kg_context(context):
                continue

            score = _get_context_value(context, "score")
            parsed_triple = _parse_kg_triple(_get_context_value(context, "text", ""))
            if parsed_triple is None:
                continue

            subject, predicate, obj = parsed_triple
            parsed_edges.append((subject, predicate, obj, score))

    except Exception as e:
        logger.error(f"Error converting contexts to nodes: {e}", exc_info=True)

    return build_graph_nodes_from_edges(parsed_edges)


def _get_context_value(context: Any, key: str, default=None):
    """Return *key* from *context*, supporting both dict and object access."""
    if isinstance(context, dict):
        return context.get(key, default)
    return getattr(context, key, default)


def _is_kg_context(context: Any) -> bool:
    """Return ``True`` if *context* represents a knowledge-graph triple."""
    context_type = _get_context_value(context, "context_type")
    if isinstance(context_type, str):
        return context_type == "knowledge_graph"
    return getattr(context_type, "value", None) == "knowledge_graph" or str(context_type).endswith(".KG")


def _truncate_label(text: str, max_length: int = 50) -> str:
    """Return *text* truncated to *max_length* characters, with an ellipsis if clipped."""
    return text[:max_length] + "..." if len(text) > max_length else text


def _parse_kg_triple(context_text: str) -> Optional[Tuple[Dict[str, str], str, Dict[str, str]]]:
    """Parse a ``(subject, predicate, object)`` triple string into its three components.

    Returns ``None`` if *context_text* is not a well-formed triple.
    """
    triple_text = context_text.strip()
    if not triple_text.startswith("(") or not triple_text.endswith(")"):
        return None

    parts = [part.strip() for part in triple_text[1:-1].split(",", 2)]
    if len(parts) != 3:
        return None

    subject = _parse_entity(parts[0], default_type="resource")
    obj = _parse_entity(parts[2], default_type="literal")
    return subject, parts[1], obj


def _parse_entity(raw_value: str, default_type: str) -> Dict[str, str]:
    """Parse a ``<ID>:<label>`` entity string into a node dict.

    If *raw_value* matches the ``<ID>:<label>`` pattern (where the ID matches
    ``ENTITY_ID_PATTERN``), the id and label are split accordingly.  Otherwise
    the full string is used as both the id and the label.

    Args:
        raw_value: Raw entity string from the triple, e.g. ``"R42:Some Paper"``.
        default_type: Node type to assign when the ID is not present (``"resource"``
            or ``"literal"``).
    """
    identifier, separator, label = raw_value.partition(":")
    candidate_id = identifier.strip()

    if separator and ENTITY_ID_PATTERN.match(candidate_id):
        return {
            "id": candidate_id,
            "label": label.strip() or candidate_id,
            "type": "literal" if candidate_id.startswith("L") else "resource",
        }

    cleaned_value = raw_value.strip()
    return {
        "id": cleaned_value,
        "label": cleaned_value,
        "type": default_type,
    }


def _get_or_create_entity_node(
    nodes_by_id: Dict[str, GraphNode],
    entity: Dict[str, str],
    score: Optional[float],
) -> GraphNode:
    """Return the ``GraphNode`` for *entity*, creating it if it does not yet exist.

    If a node for the entity id already exists, its ``score`` is updated when
    *score* is higher, and its ``label`` is filled in if it was previously empty.

    Args:
        nodes_by_id: Mutable id→node mapping shared across all edges.
        entity: Dict with ``"id"``, ``"label"``, and ``"type"`` keys.
        score: Relevance score for this occurrence; may be ``None``.
    """
    existing_node = nodes_by_id.get(entity["id"])

    if existing_node is None:
        existing_node = GraphNode(
            id=entity["id"],
            label=entity["label"],
            type=entity["type"],
            connections=[],
            score=score,
        )
        nodes_by_id[entity["id"]] = existing_node
        return existing_node

    if score is not None and (existing_node.score is None or score > existing_node.score):
        existing_node.score = score

    if existing_node.label == existing_node.id and entity["label"]:
        existing_node.label = entity["label"]

    return existing_node


def _extract_sources(contexts) -> List[str]:
    """
    Extract unique source identifiers from contexts.

    Args:
        contexts: List of Context objects.

    Returns:
        Deduplicated list of source identifiers (DOIs, URLs, etc.).
    """
    sources = []
    try:
        for context in contexts:
            if context.metadata:
                if "source_doi" in context.metadata:
                    sources.append(context.metadata["source_doi"])
                elif "source" in context.metadata:
                    sources.append(context.metadata["source"])
                elif "doi" in context.metadata:
                    sources.append(context.metadata["doi"])
    except Exception as e:
        logger.error(f"Error extracting sources: {e}")

    return list(set(sources))
