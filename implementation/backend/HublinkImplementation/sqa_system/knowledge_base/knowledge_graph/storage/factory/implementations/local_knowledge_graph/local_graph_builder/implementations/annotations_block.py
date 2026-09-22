from typing_extensions import override
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, RDFS

from sqa_system.core.data.models.publication import Publication
from ......implementations.local_knowledge_graph import (
    LocalKnowledgeGraph,
    RE,
    EX,
    XSD
)

from ..base.publication_rdf_graph_block import PublicationRDFGraphBlock


class AnnotationsBlock(PublicationRDFGraphBlock):
    """
    A block for adding annotations to a publication node in a knowledge graph.
    """

    @override
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):
        if not publication.additional_fields:
            return

        if "annotations" in publication.additional_fields:
            self._add_annotations_to_publication(graph,
                                                 publication.additional_fields["annotations"],
                                                 pub_uri)

    def _add_annotations_to_publication(self,
                                        graph: LocalKnowledgeGraph,
                                        annotations: dict,
                                        pub_uri: URIRef):
        """
        As part of our master thesis we are using a specific set of annotations
        to describe the content of a publication based on expert knowledge.
        This method adds these annotations to the graph.

        Args:
            graph: The graph to build the annotations block in.
            annotations: The dictionary containing the annotations.
            pub_uri: The URI of the publication node.
        """
        for key, value in annotations.items():
            if key == "Meta Data":
                for meta_key, meta_value in value.items():
                    add_literal(pub_uri, meta_key, meta_value, graph)
            elif key == "Validity":
                self._add_validity(graph, value, pub_uri)
            elif "Research Object" in key:
                obj_uri = RE[graph.get_new_id('researchObject')]
                graph.graph.add((obj_uri, RDF.type, EX.ResearchObject))
                graph.graph.add((pub_uri, EX.hasResearchObject, obj_uri))
                if not isinstance(value, dict):
                    add_literal(obj_uri, key, value, graph)
                    continue
                for obj_key, obj_value in value.items():
                    if obj_key == "Research Object":
                        graph.graph.add(
                            (obj_uri, RDFS.label, Literal(obj_value)))
                        continue
                    if "Evaluation" in obj_key:
                        self._add_evaluation(
                            graph, obj_uri, obj_value, obj_key)

    def _add_evaluation(self,
                        graph: LocalKnowledgeGraph,
                        obj_uri: URIRef,
                        obj_value: dict,
                        obj_key: str):
        """
        Adds the evaluation annotation to the graph.

        Args:
            graph: The graph to build the evaluation block in.
            obj_uri: The URI of the research object node.
            obj_value: The dictionary containing the evaluation data.
            obj_key: The key for the evaluation data.
        """
        eval_uri = RE[graph.get_new_id('evaluation')]
        graph.graph.add((eval_uri, RDF.type, EX.Evaluation))
        graph.graph.add((obj_uri, EX.hasEvaluation, eval_uri))
        if not isinstance(obj_value, dict):
            add_literal(eval_uri, obj_key, obj_value, graph)
            return
        for eval_key, eval_value in obj_value.items():
            if eval_key == "Properties":
                self._add_properties(graph, eval_uri, eval_value)
            else:
                add_literal(eval_uri, eval_key, eval_value, graph)

    def _add_validity(self,
                      graph: LocalKnowledgeGraph,
                      value: dict,
                      pub_uri: URIRef):
        """
        Adds the validity annotation to the graph.

        Args:
            graph: The graph to build the validity block in.
            value: The dictionary containing the validity data.
            pub_uri: The URI of the publication node.
        """
        val_uri = RE[graph.get_new_id('validity')]
        graph.graph.add((val_uri, RDF.type, EX.Validity))
        graph.graph.add((pub_uri, EX.hasValidity, val_uri))
        if value == "Replication Package":
            graph.graph.add(
                (val_uri, EX.hasReplicationPackage, Literal(True, datatype=XSD.boolean)))
            return
        if value == "Referenced Threats To Validity Guideline":
            graph.graph.add((val_uri, EX.hasReferencedThreatsToValidityGuideline, Literal(
                True, datatype=XSD.boolean)))
            return
        for val_key, val_value in value.items():
            if val_key == "Threats to Validity":
                for threat in val_value:
                    graph.graph.add(
                        (val_uri, EX.hasThreat, Literal(threat)))
                continue
            add_literal(val_uri, val_key, val_value, graph)

    def _add_properties(self,
                        graph: LocalKnowledgeGraph,
                        eval_uri: URIRef,
                        eval_value: dict):
        """
        Adds the properties annotation to the graph.

        Args:
            graph: The graph to build the properties block in.
            eval_uri: The URI of the evaluation node.
            eval_value: The dictionary containing the properties data.
        """
        prop_uri = RE[graph.get_new_id('property')]
        graph.graph.add(
            (prop_uri, RDF.type, EX.Property))
        graph.graph.add(
            (eval_uri, EX.hasProperty, prop_uri))
        properties = set()
        for prop_key, prop_value in eval_value.items():
            properties.add(prop_key)
            if isinstance(prop_value, list):
                for prop in prop_value:
                    prop = prop.replace(" ", "").replace(
                        "[", "").replace("]", "").replace("'", "")
                    graph.graph.add(
                        (prop_uri, EX["examines"], Literal(prop)))
            else:
                prop_value = prop_value.replace(" ", "").replace(
                    "[", "").replace("]", "").replace("'", "")
                graph.graph.add(
                    (prop_uri, EX["examines"], Literal(prop_value)))
        for prop_key in properties:
            graph.graph.add(
                (prop_uri, RDFS.label, Literal(prop_key)))


def add_literal(uri: URIRef,
                key: str,
                value: str,
                graph: LocalKnowledgeGraph):
    """
    Helper function to add a literal to the graph.

    Args:
        uri: The URI of the node to which the literal will be added.
        key: The key for the literal.
        value: The value of the literal.
        graph: The graph to which the literal will be added.
    """
    key = key.replace(" ", "")
    key_uri = EX[f"has{key}"]
    graph.graph.add((uri, key_uri, Literal(value)))
