from typing_extensions import override
from rdflib import Literal, URIRef
from rdflib.namespace import RDFS, XSD

from sqa_system.core.data.models.publication import Publication
from ......implementations.local_knowledge_graph import (
    LocalKnowledgeGraph,
    SCHEMA,
    PRE
)
from ..base.publication_rdf_graph_block import PublicationRDFGraphBlock


class MetadataBlock(PublicationRDFGraphBlock):
    """
    A block for adding metadata to a publication node in a knowledge graph.
    """

    @override
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):

        graph.graph.add(
            (pub_uri, RDFS.label, Literal(publication.title)))

        graph.graph.add((pub_uri, PRE["doi"],
                         Literal(publication.doi)))

        if publication.url:
            graph.graph.add((pub_uri, SCHEMA.url, Literal(
                publication.url, datatype=XSD.anyURI)))

        if publication.year:
            if publication.month:
                date_literal = Literal(
                    f"{publication.year}-{publication.month}")
            else:
                date_literal = Literal(publication.year)
            graph.graph.add((pub_uri, SCHEMA.datePublished, date_literal))

        if publication.abstract:
            graph.graph.add((pub_uri, SCHEMA.abstract,
                            Literal(publication.abstract)))

        if publication.keywords:
            for keyword in publication.keywords:
                graph.graph.add(
                    (pub_uri, SCHEMA.keywords, Literal(keyword)))
