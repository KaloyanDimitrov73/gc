from abc import ABC, abstractmethod
import urllib.parse
from rdflib import URIRef

from sqa_system.core.data.models.publication import Publication
from ......implementations.local_knowledge_graph import LocalKnowledgeGraph


class PublicationRDFGraphBlock(ABC):
    """
    A block interface for building specific parts of a knowledge graph 
    based on a publication.
    """
    @abstractmethod
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):
        """
        Builds the block for a publication node in the graph.

        Args:
            graph: The graph to build the block in.
            pub_uri: The URI of the publication node.
            publication: The publication to build the block for.
        """

    def _sanitize_identifier(self, value: str) -> str:
        """
        Sanitizes the identifier by replacing spaces with underscores and
        encoding it for use in a URI.
        
        Args:
            value: The identifier to sanitize.
            
        Returns:
            str: The sanitized identifier.
        """
        value = value.replace(" ", "_")
        return urllib.parse.quote(value.strip(), safe="/:")
