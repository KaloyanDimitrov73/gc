from typing import List, Optional, Sequence

from core.data.models import Knowledge
from core.logging.logging import get_logger
from knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph

logger = get_logger(__name__)


def resolve_root_entities(
    graph: KnowledgeGraph,
    root_entity_ids: Optional[List[str]] = None,
    root_entity_types: Optional[List[str]] = None,
) -> List[Knowledge]:
    """
    Resolves root entities for indexing from a graph, based on explicit
    ids and/or entity types.

    Args:
        graph (KnowledgeGraph): The graph to resolve entities against.
        root_entity_ids (List[str], optional): Explicit root entity ids.
        root_entity_types (List[str], optional): Root entity types

    Returns:
        List[Knowledge]: The union of entities matched by id and by type.

    Raises:
        ValueError: if neither root_entity_ids nor root_entity_types is given.
    """
    if not root_entity_ids and not root_entity_types:
        raise ValueError(
            "Either root_entity_types or root_entity_ids must be specified."
        )

    root_entities: List[Knowledge] = []

    if root_entity_ids:
        for root_entity_id in root_entity_ids:
            root_entity = graph.get_entity_by_id(root_entity_id)
            if root_entity:
                root_entities.append(root_entity)
            else:
                logger.warning("Root entity with ID %s not found", root_entity_id)

    if root_entity_types:
        root_entities.extend(graph.get_entities_by_types(set(root_entity_types)))

    if not root_entities:
        logger.warning(
            "No root entities found for indexing. Types: %s, Ids: %s",
            root_entity_types,
            root_entity_ids,
        )

    return root_entities