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


class ResearchFieldBlock(PublicationRDFGraphBlock):
    """
    A block for adding a research field to a publication node in a knowledge graph.
    """

    @override
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):
        if not publication.research_field:
            return

        field_uri = self._get_or_create_research_field(
            graph, publication.research_field)
        graph.graph.add((pub_uri, PRE.research_field, field_uri))

    def _get_or_create_research_field(self, graph: LocalKnowledgeGraph, field_name: str) -> URIRef:
        """
        This method checks if a research field already exists in the graph.
        If it does, it returns the existing research field URI.
        If it doesn't, it creates a new research field URI and adds it to the graph.

        Args:
            graph: The graph to check for the research field.
            field_name: The name of the research field to check for.
        Returns:
            URIRef: The URI of the research field.
        """
        query = f"""
            SELECT ?field WHERE {{
                ?field a class:research_field ;                
                    rdfs:label "{field_name}" .
            }}
        """
        results = graph.run_sparql_query(query)
        if not results.empty:
            uri_str = results.iloc[0]['field']
            return URIRef(uri_str)

        # If not research field has been found, we create a new one
        field_uri = RE[graph.get_new_id("research_field")]
        graph.graph.add((field_uri, RDF.type, CLASS.research_field))
        graph.graph.add((field_uri, RDFS.label, Literal(field_name)))
        return field_uri
