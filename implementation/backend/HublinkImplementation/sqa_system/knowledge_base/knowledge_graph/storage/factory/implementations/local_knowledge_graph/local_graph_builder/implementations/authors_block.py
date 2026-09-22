from typing_extensions import override
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, RDFS
from pylatexenc.latex2text import LatexNodes2Text

from sqa_system.core.data.models.publication import Publication
from ......implementations.local_knowledge_graph import (
    LocalKnowledgeGraph,
    RE,
    PRE,
    SCHEMA
)
from ..base.publication_rdf_graph_block import PublicationRDFGraphBlock


class AuthorsBlock(PublicationRDFGraphBlock):
    """
    A block for adding authors to a publication node in a knowledge graph.
    """
    
    
    def _correct_author_name(self, author_name: str) -> str:
        """
        We encountered some latex encodings in the data such as:
        Dominik Fuch{\\ss
        Marek G{\\'{o}}rski
        Javier C{\\'{a}}mara
        Sten Gr{\\\"{u}}ner
        Julius R{\\\"{u}}ckert
        
        This method corrects these encodings to the correct form.
        
        Args: 
            author_name (str): The author name to correct.
            
        Returns:
            str: The corrected author name.
        """
        return LatexNodes2Text().latex_to_text(author_name)

    @override
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):
        if not publication.authors:
            return

        # Create a unique URI for the list of authors
        author_list_id = RE[graph.get_new_id("authors")]
        graph.graph.add((pub_uri, PRE["authors"], author_list_id))

        # Add authors
        for index, author_name in enumerate(publication.authors, start=1):
            corrected_author_name = self._correct_author_name(author_name)
            author_uri = self._get_or_create_author(graph, corrected_author_name)
            graph.graph.add(
                (author_list_id, PRE[f"entry_{index}"], author_uri))

    def _get_or_create_author(self,
                              graph: LocalKnowledgeGraph,
                              author_name: str) -> URIRef:
        """
        Checks if an author exists in the graph and creates one if not.
        
        Args:
            graph: The graph to check for the author.
            author_name: The name of the author to find or create.
        
        Returns:
            URIRef: The URI of the author.
        """
        # Check if the author already exists in the graph
        query = f"""
            SELECT ?author WHERE {{
                ?author a schema:Person ;
                    rdfs:label "{author_name}" .
            }}
        """
        results = graph.run_sparql_query(query)
        if not results.empty:
            author_uri_str = results.iloc[0]['author']
            return URIRef(author_uri_str)

        # If not found, create a new author
        author_uri = RE[graph.get_new_id("author")]
        graph.graph.add((author_uri, RDF.type, SCHEMA.Person))
        graph.graph.add((author_uri, RDFS.label, Literal(author_name)))
        return author_uri
