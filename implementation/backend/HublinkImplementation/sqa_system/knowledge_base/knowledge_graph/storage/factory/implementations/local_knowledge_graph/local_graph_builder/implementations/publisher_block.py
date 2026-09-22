from typing_extensions import override
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, RDFS

from sqa_system.core.data.models.publication import Publication
from ......implementations.local_knowledge_graph import (
    LocalKnowledgeGraph,
    RE,
    PRE,
    CLASS
)
from ..base.publication_rdf_graph_block import PublicationRDFGraphBlock


class PublisherBlock(PublicationRDFGraphBlock):
    """
    A block for adding a publisher to a publication node in a knowledge graph.
    """

    @override
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):
        if not publication.publisher:
            return

        publisher_uri = self._get_or_create_publisher(
            graph, publication.publisher)
        graph.graph.add((pub_uri, PRE["publisher"], publisher_uri))

    def _get_or_create_publisher(self,
                                 graph: LocalKnowledgeGraph,
                                 publisher_name: str) -> URIRef:
        """
        This method checks if a publisher already exists in the graph.
        If it does, it returns the existing publisher URI.
        If it doesn't, it creates a new publisher URI and adds it to the graph.
        
        Args:
            graph: The graph to check for the publisher.
            publisher_name: The name of the publisher to check for.
        Returns:
            URIRef: The URI of the publisher.
        """
        query = f"""
            SELECT ?publisher WHERE {{
                ?publisher a class:publisher ;
                    rdfs:label "{publisher_name}" .
            }}
        """
        results = graph.run_sparql_query(query)
        if not results.empty:
            uri_str = results.iloc[0]['publisher']
            return URIRef(uri_str)

        # If not publisher has been found, we create a new one
        publisher_uri = RE[graph.get_new_id("publisher")]
        graph.graph.add((publisher_uri, RDF.type, CLASS.publisher))
        graph.graph.add((publisher_uri, RDFS.label, Literal(publisher_name)))
        return publisher_uri
