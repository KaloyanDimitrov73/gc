from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from sqa_system.core.data.models.triple import Triple
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models.knowledge import Knowledge
from .hub_path import HubPath
from .entity_with_direction import EntityWithDirection

class Hub(BaseModel):
    """
    Represents a 'Hub' in the knowledge graph. A hub is a special concept used
    in the retrieval approach that is of particular significance for a specific 
    domain or research question. Hubs consolidate information and serve as pivotal 
    points for searching relevant data
    """
    root_entity: EntityWithDirection = Field(...,
                                             description="The root entity of the hub.")
    paths: List[HubPath] = Field(
        ..., description="The paths of the hub to either the end or the next hub.")
    hub_score: Optional[float] = Field(None, description="The relevance score of the hub.")

    @staticmethod
    def is_hub_entity(entity: Knowledge,
                      graph: KnowledgeGraph,
                      options: 'IsHubOptions') -> bool:
        """
        Wether the given entity is classified as a hub or not.

        The function classifies an entity as a hub if it has more than
        a given amount of edges.
        """

        types = options.types
        # If the entity is not a valid id (e.g. if its a Literal), it can't be a hub
        if (entity.uid is None or not graph.is_intermediate_id(entity.uid)):
            return False

        # We are using two ways to check for the type. Some graphs directly store the
        # type of entities with the entity (e.g. the ORKG), while others store the type
        # in the relations (e.g. RDF graphs). We check both ways.
        includes_type = False
        tail_relations = graph.get_relations_of_head_entity(entity)
        head_relations = graph.get_relations_of_tail_entity(entity)
        if isinstance(types, list) and isinstance(types[0], str):
            includes_type = any(
                type in types for type in entity.knowledge_types)
        elif isinstance(types, list) and isinstance(types[0], tuple):
            includes_type = check_relations_for_type(tail_relations, types)
            if not includes_type:
                includes_type = check_relations_for_type(head_relations, types)
        else:
            includes_type = False

        # If the amount of edges is not specified, we only check for the type
        if options.hub_edges == -1:
            return includes_type

        # If the amount of edges is specified, we also check if the entity has enough edges
        has_enough_edges = len(tail_relations) + len(head_relations) >= options.hub_edges
        return includes_type and has_enough_edges


class IsHubOptions(BaseModel):
    """
    Options for the IsHub function.

    With this class, we can specify what entities are classified as a hub.
    """
    hub_edges: int = Field(
        -1, description="The amount of edges an entity needs to have to be classified as a hub.")
    types: Optional[List[Tuple[str, str] | str]] = Field(
        None,
        description=("The types of entities that are considered as hubs. "
                     "Input can be a list of strings or a list of tuples "
                     "where the first element is the predicate and the second "
                     "element is the object of the relation."))


def check_relations_for_type(relations: List[Triple], types: List[Tuple[str, str]]) -> bool:
    """
    Checks if the given relations contain a relation with the given types.
    """
    for relation in relations:
        for type_tuple in types:
            if relation.predicate == type_tuple[0] and relation.entity_object.text == type_tuple[1]:
                return True
    return False
