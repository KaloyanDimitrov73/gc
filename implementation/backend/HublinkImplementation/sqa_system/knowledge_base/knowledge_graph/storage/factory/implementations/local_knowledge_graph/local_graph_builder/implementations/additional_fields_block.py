from typing_extensions import override
from rdflib import Literal, URIRef
from rdflib.namespace import RDFS

from sqa_system.core.data.models.publication import Publication
from ......implementations.local_knowledge_graph import (
    LocalKnowledgeGraph,
    RE,
    PRE
)
from ..base.publication_rdf_graph_block import PublicationRDFGraphBlock


class AdditionalFieldsBlock(PublicationRDFGraphBlock):
    """
    A block for adding additional fields to a publication node in a knowledge graph.
    """

    @override
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):
        if not publication.additional_fields:
            return

        for key, value in publication.additional_fields.items():
            if key == "annotations":
                continue
            key_uri = PRE[f"{self._sanitize_identifier(key)}"]
            root_uri = RE[graph.get_new_id(self._sanitize_identifier(key))]
            graph.graph.add((root_uri, RDFS.label, Literal(key)))
            if isinstance(value, dict):
                value = self._add_dictionary(graph, value, root_uri)
            else:
                value = Literal(value)
            graph.graph.add((pub_uri, key_uri, value))

    def _add_dictionary(self,
                        graph: LocalKnowledgeGraph,
                        dictionary: dict,
                        root_uri: URIRef) -> URIRef:
        """
        Builds a dictionary block in the knowledge graph.

        Args:
            graph: The graph to build the dictionary block in.
            dictionary: The dictionary containing additional fields.
            root_uri: The URI of the root node for the dictionary.
            
        Returns:
            URIRef: The URI of the root node for the dictionary block.
        """
        for key, value in dictionary.items():
            key_uri = PRE[f"{self._sanitize_identifier(key)}"]
            if value is None:
                continue
            if isinstance(value, dict):
                new_root_uri = RE[graph.get_new_id(self._sanitize_identifier(key))]
                graph.graph.add((root_uri, key_uri, new_root_uri))
                value = self._add_dictionary(graph, value, new_root_uri)
            else:
                value = Literal(value)
            graph.graph.add((root_uri, key_uri, value))
        return root_uri
