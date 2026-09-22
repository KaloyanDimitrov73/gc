from abc import ABC, abstractmethod
from typing import List, Set
from rdflib.namespace import RDF
from rdflib.plugins.sparql.parser import parseQuery

from sqa_system.core.config.models.knowledge_base.knowledge_graph_config import KnowledgeGraphConfig
from sqa_system.core.data.models.triple import Triple
from sqa_system.core.data.models.knowledge import Knowledge


class KnowledgeGraph(ABC):
    """
    The main interface for Knowledge Graphs in the QA system.
    Each knowledge graph implementation should inherit from this class.
    """

    @property
    def root_type(self) -> str:
        """
        The identifier in the graph that is used for indicating the type of
        an entity.

        Returns:    
            str: The identifier in the graph that is used for indicating the type of
            an entity. Defaults to RDF.type.
        """
        return str(RDF.type)

    @property
    @abstractmethod
    def paper_type(self) -> str:
        """
        The identifier that is used to identify a node as a paper.

        Returns:
            str: The identifier that is used to identify a node as a paper.
        """

    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config

    @abstractmethod
    def get_random_publication(self) -> Knowledge:
        """
        Returns a random publication from the knowledge graph.

        Returns:
            Knowledge: A random publication from the knowledge graph.
        """

    @abstractmethod
    def is_publication_allowed_for_generation(self, publication: Knowledge) -> bool:
        """
        Returns whether the publication should be used for generation.

        Args:
            publication (Knowledge): The publication to check.

        Returns:
            bool: True if the publication should be used for generation, 
                False otherwise.
        """

    @abstractmethod
    def validate_graph_connection(self) -> bool:
        """
        Checks if it is possible to query on the knowledge graph.

        Returns:
            bool: True if the connection is valid, False otherwise.
        """

    @abstractmethod
    def get_main_triple_from_publication(self, publication_id: str) -> Triple | None:
        """
        Returns the main triple of a publication in the knowledge graph.

        Args:
            publication_id (str): The ID of the entity.

        Returns:
            Triple: The main triple of the entity.

        Example:
            >>> get_main_triple_from_entity_id("Q123")
            >>> Triple(entity_subject=Knowledge(text="Q123"), 
                        entity_object=Knowledge(text="Entity Name"), relation="rdfs:label")
        """

    @abstractmethod
    def get_relations_of_tail_entity(self, knowledge: Knowledge) -> List[Triple]:
        """
        Returns all head relations of an entity in the knowledge graph.
        A head relation is a relation where the knowledge id is on the head
        of the triple

        Args:
            knowledge (Knowledge): The knowledge entity to get the relations of

        Returns:
            List[Triple]: A list of triples that are the relations of the entity.

        Example:
            >>> get_relations_of_tail_entity(Knowledge(uid="Q123"))
            >>> [Triple(entity_subject=Knowledge(uid="SOMETHING_ELSE"), 
                    entity_object=Knowledge(uid="Q123"), relation="P31")]
        """

    @abstractmethod
    def get_relations_of_head_entity(self, knowledge: Knowledge) -> List[Triple]:
        """
        Returns all tail relations of an entity in the knowledge graph.
        A tail relation is a relation where the knowledge id is on the tail
        of the triple e.g. (other_knowledge_id, relation, knowledge_id).

        Args:
            knowledge (Knowledge): The knowledge entity to get the relations of

        Returns:
            List[Triple]: A list of triples that are the relations of the entity.

        Example:
            >>> get_relations_of_head_entity(Knowledge(uid="Q123"))
            >>> [Triple(entity_subject=Knowledge(uid="Q123"), 
                    entity_object=Knowledge(uid="SOMETHING_ELSE"), relation="P31")]

        """

    @abstractmethod
    def is_intermediate_id(self, entity_id: str) -> bool:
        """
        Returns whether the given id is a valid knowledge id that is not yet a leaf
        in the knowledge graph.

        Args:
            entity_id (str): The ID of the entity.

        Returns:
            bool: True if the ID is valid, False otherwise.
        """

    @abstractmethod
    def get_entities_by_predicate_id(self, predicates: Set[str]) -> Set[Knowledge]:
        """
        Retrieves entities that have the specified predicates.

        Args:
            predicates (Set[str]): The predicates to search for.

        Returns:
            Set[Knowledge]: A set of entities that have the specified predicates.
        """

    @abstractmethod
    def get_entity_ids_by_types(self, types: Set[str]) -> Set[str]:
        """
        Retrieves entities that are of the specified RDF types.

        Args:
            types (Set[str]): The RDF types to search for.

        Returns:
            Set[str]: A set of entity IDs that match the specified types.
        """

    @abstractmethod
    def get_types_of_entity(self, entity: Knowledge) -> Set[str]:
        """
        Retrieves the types of an entity.

        Args:
            entity (Knowledge): The entity to get the types of.

        Returns:
            Set[str]: A set of types associated with the entity.
        """

    def get_entities_by_types(self, types: Set[str]) -> Set[Knowledge]:
        """
        Retrieves entities that are of the specified RDF types.

        Args:
            types (Set[str]): The RDF types to search for.

        Returns:
            Set[Knowledge]: A set of entities that match the specified types.
        """
        entity_ids = self.get_entity_ids_by_types(types)
        entities = set()
        for entity_id in entity_ids:
            entity = self.get_entity_by_id(entity_id)
            if entity is not None:
                entities.add(entity)
        return entities

    @abstractmethod
    def get_entity_by_id(self, entity_id: str) -> Knowledge | None:
        """
        Retrieves an entity by its ID. None if the entity is not in the graph.

        Args:
            entity_id (str): The ID of the entity.

        Returns:
            Knowledge | None: The entity if found, None otherwise.
        """

    def get_all_triples_from_the_end_of_the_path(self, current: Knowledge) -> List[Triple]:
        """
        Traverses the edges of the knowledge entity 
        until the end of the path is reached.

        Returns all the values gathered.

        Args:
            current (Knowledge): The current knowledge entity to traverse.

        Returns:
            List[Triple]: A list of triples at the leaf of the path.
        """
        if current is None or self.is_intermediate_id(current.uid) is False:
            return []
        relations = self.get_relations_of_head_entity(current)
        values: List[Triple] = []
        for relation in relations:
            if relation.entity_object is None:
                continue
            if self.is_intermediate_id(relation.entity_object.uid):
                # We need to further traverse the path
                values.extend(self.get_all_triples_from_the_end_of_the_path(
                    relation.entity_object))
            else:
                values.append(relation)
        return values

    def get_paper_from_entity(self,
                              entity: Knowledge) -> Knowledge | None:
        """
        Given an entity, this function traverses the graph backwards to find
        the first paper entity and returns it.

        Args:
            entity (Knowledge): The entity to traverse from.

        Returns:
            Knowledge | None: The first paper entity found, or None if not found.
        """
        if entity is None:
            return None

        relations = self.get_relations_of_head_entity(entity)

        # Check if the given entity is the root entity
        for relation in relations:
            # First check if the knowledge entity contains the type
            if (relation.entity_object.knowledge_types is not None and
                    self.paper_type in relation.entity_object.knowledge_types):
                return relation.entity_object
            if (relation.entity_subject.knowledge_types is not None and
                    self.paper_type in relation.entity_subject.knowledge_types):
                return relation.entity_subject
            # Check if the type is defined per predicate
            if (str(RDF.type) in relation.predicate
                    and self.paper_type in relation.entity_object.text):
                return relation.entity_subject

        # Else recursively traverse the graph backwards
        parent_relations = self.get_relations_of_tail_entity(entity)
        for relation in parent_relations:
            parent = relation.entity_subject
            result = self.get_paper_from_entity(parent)
            if result is not None:
                return result

        return None

    def validate_sparql_query(self, query: str) -> bool:
        """
        Validates a SPARQL query.

        Args:
            query (str): The SPARQL query to validate.

        Returns:
            bool: True if the query is valid, False otherwise.
        """
        try:
            parseQuery(query)
        except Exception:
            return False
        return True
