import urllib.parse
from typing import List
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, RDFS
from sqa_system.core.data.models.publication import Publication
from sqa_system.knowledge_base.knowledge_graph.storage.implementations.local_knowledge_graph import (
    LocalKnowledgeGraph,
    PREFIX_TO_NAMESPACE,
    RE,
    SCHEMA,
    PRE,
    CLASS
)
from sqa_system.core.logging.logging import get_logger

from .base.publication_rdf_graph_block import PublicationRDFGraphBlock

logger = get_logger(__name__)


class RDFLIBGraphBuilder:
    """
    Class responsible for the construction of a rdflib graph.
    """

    def __init__(self):
        self.publication_blocks: List[PublicationRDFGraphBlock] = []

    def initialize_graph(self, graph: LocalKnowledgeGraph):
        """
        Initializes the graph with the specified configuration.

        Args:
            graph: The graph to be initialized.
        """
        self._initialize_namespaces(graph)
        self._initialize_base_triples(graph)
        logger.info("A new rdflib graph has been initialized.")

    def add_publication_block(self, block: PublicationRDFGraphBlock):
        """
        Adds a publication block to the builder.

        Args:
            block: The publication block to be added.
        """
        self.publication_blocks.append(block)

    def build_publication_node(self,
                               graph: LocalKnowledgeGraph,
                               publication: Publication) -> URIRef:
        """
        Builds a publication node in the graph.
        It uses the publication blocks to add data to the node.

        Args: 
            graph: The graph to build the publication node in.
            publication: The publication to build the node for.

        Returns:
            URIRef: The URI of the publication node.
        """
        pub_uri = self._generate_pub_uri(publication)
        graph.graph.add((pub_uri, RDF.type, SCHEMA.ScholarlyArticle))

        # Add data to the publication node
        for block in self.publication_blocks:
            block.build(graph, pub_uri, publication)

        return pub_uri

    def _generate_pub_uri(self, publication: Publication) -> URIRef:
        """
        Generates a URI for the publication based on its DOI.

        Args:
            publication: The publication to generate the URI for.

        Returns:
            URIRef: The generated URI for the publication.
        """
        return RE[f"publication/{self._sanitize_identifier(publication.doi)}"]

    def _sanitize_identifier(self, value: str) -> str:
        """
        Sanitizes the identifier by replacing spaces with underscores and encoding it.

        Args:
            value: The identifier to sanitize.

        Returns:
            str: The sanitized identifier.
        """
        value = value.replace(" ", "_")
        return urllib.parse.quote(value.strip(), safe="/:")

    def _initialize_namespaces(self, graph: LocalKnowledgeGraph):
        """
        Initializes the namespaces for the graph.

        Args:
            graph: The graph to initialize the namespaces for.
        """
        for prefix, namespace in PREFIX_TO_NAMESPACE.items():
            graph.graph.bind(prefix, namespace)

    def _initialize_base_triples(self, graph: LocalKnowledgeGraph):
        """
        Initializes the base triples for the graph with classes and predicates.

        Args:
            graph: The graph to initialize the base triples for.
        """
        # Add classes to the graph
        graph.graph.add((CLASS["venue"], RDF.type, RDFS.Class))
        graph.graph.add((CLASS["research_field"], RDF.type, RDFS.Class))
        graph.graph.add((CLASS["publisher"], RDF.type, RDFS.Class))

        # Add predicates to the graph
        graph.graph.add(
            (PRE["research_field"], RDF.type, Literal("predicate")))
        graph.graph.add(
            (PRE["research_field"], RDFS.label, Literal("research field")))

        graph.graph.add((PRE["authors"], RDF.type, Literal("predicate")))
        graph.graph.add((PRE["authors"], RDFS.label, Literal("authors")))

        graph.graph.add((PRE["author"], RDF.type, Literal("predicate")))
        graph.graph.add((PRE["author"], RDFS.label, Literal("author")))

        graph.graph.add((PRE["doi"], RDF.type, Literal("predicate")))
        graph.graph.add((PRE["doi"], RDFS.label, Literal("doi")))

        graph.graph.add((PRE["venue"], RDF.type, Literal("predicate")))
        graph.graph.add((PRE["venue"], RDFS.label, Literal("venue")))
