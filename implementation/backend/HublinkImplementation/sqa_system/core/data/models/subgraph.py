from typing import List, Set, Optional

from pydantic import RootModel, Field

from .triple import Triple
from .knowledge import Knowledge


class Subgraph(RootModel):
    """
    Represents a subgraph in a Knowledge Graph consisting of a list of triples.
    """
    root: List[Triple] = Field(
        default_factory=list,
        description="The list of triples that form the subgraph.")

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]
    
    def __len__(self):
        return len(self.root)

    def get_entities_by_id(self,
                            entity_id_substring: str) -> Set[Knowledge]:
        """
        Retrieves all entities that are of type entity_type_substring from the subgraph.

        Args:
            entity_type_substring (str): The type of the golden entity. Can be a 
                substring of the entity type.
                
        Returns:
            Set[Knowledge]: A set of Knowledge objects that are of the specified type.
        """
        golden_entities = set()
        for triple in self.root:
            if entity_id_substring in triple.entity_object.uid:
                golden_entities.add(triple.entity_object)
            if entity_id_substring in triple.entity_subject.uid:
                golden_entities.add(triple.entity_subject)
        return golden_entities

    def get_triples_with_predicate(self,
                                    predicate_substring: str,
                                    value_substring: Optional[str] = None) -> List[Triple]:
        """
        Retrieves all triples that contain a specified predicate and value substring.

        Args:
            subgraph (List[Triple]): The subgraph to search for triples.
            predicate_substring (str): The predicate substring to search for.
            value_substring (Optional[str]): An additional option to only include triples
                that contain a specific value substring.
                
        Returns:
            List[Triple]: A list of triples that match the specified predicate and value.
        """
        triples = []
        for triple in self.root:
            if predicate_substring in triple.predicate:
                if not value_substring:
                    triples.append(triple)
                    continue
                if value_substring in triple.entity_object.text:
                    triples.append(triple)
                if value_substring in triple.entity_subject.text:
                    triples.append(triple)
        return triples
