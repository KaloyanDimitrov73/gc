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


class VenueBlock(PublicationRDFGraphBlock):
    """
    A block for adding a venue to a publication node in a knowledge graph.
    """

    @override
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):
        if not publication.venue:
            return

        venue_uri = self._get_or_create_venue(graph, publication.venue)
        graph.graph.add((pub_uri, PRE["venue"], venue_uri))

    def _get_or_create_venue(self, graph: LocalKnowledgeGraph, venue_name: str) -> URIRef:
        """
        A method that checks if a venue already exists in the graph.
        If it does, it returns the existing venue URI.
        If it doesn't, it creates a new venue URI and adds it to the graph.
        Args:
            graph: The graph to check for the venue.
            venue_name: The name of the venue to check for.
        Returns:
            URIRef: The URI of the venue.
        """
        query = f"""
            SELECT ?venue WHERE {{
                ?venue a class:venue ;                
                    rdfs:label "{venue_name}" .
            }}
        """
        results = graph.run_sparql_query(query)
        if not results.empty:
            uri_str = results.iloc[0]['venue']
            return URIRef(uri_str)

        # If not venue has been found, we create a new one
        venue_uri = RE[graph.get_new_id("venue")]
        graph.graph.add((venue_uri, RDF.type, CLASS.venue))
        graph.graph.add((venue_uri, RDFS.label, Literal(venue_name)))
        return venue_uri
