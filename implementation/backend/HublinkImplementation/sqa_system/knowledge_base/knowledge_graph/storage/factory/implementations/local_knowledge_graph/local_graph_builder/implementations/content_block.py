from typing import Union
from typing_extensions import override
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, RDFS


from sqa_system.core.data.context_tracer import ContextTrace, ContextTracer
from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.core.data.models import Publication
from sqa_system.core.data.extraction.paper_content_extractor import (
    PaperContentExtractor,
    PaperContent,
    TextWithOriginal,
    Entity
)
from sqa_system.core.logging.logging import get_logger
from ......implementations.local_knowledge_graph import (
    LocalKnowledgeGraph,
    RE,
    EX
)


from ..base.publication_rdf_graph_block import PublicationRDFGraphBlock
logger = get_logger(__name__)


class ContentBlock(PublicationRDFGraphBlock):
    """
    A block for adding content to a publication node in a knowledge graph.
    This utilizes the paper content extractor of the SQA system to use an LLM
    to extract content from the paper and then fill this content into the graph.

    Args:
        extraction_llm_config (LLMConfig): The LLM config to use for the
            paper content extractor.
    """

    def __init__(self, extraction_llm_config: LLMConfig):
        self.extraction_llm_config = extraction_llm_config
        self.tracing: ContextTracer = None
        self.current_publication: Publication = None
        self.paper_content_extractor = PaperContentExtractor(
            llm_config=extraction_llm_config)

    @override
    def build(self,
              graph: LocalKnowledgeGraph,
              pub_uri: URIRef,
              publication: Publication):
        """
        Builds the content block for a publication node in the graph.

        Args:
            graph: The graph to build the block in.
            pub_uri: The URI of the publication node.
            publication: The publication to build the block for.
        """
        self.current_publication = publication
        self.tracing = ContextTracer(
            context_id=f"content_block_{graph.config.config_hash}")

        paper_content = None
        if (self.extraction_llm_config is not None and
            publication.full_text is not None and
                publication.full_text != ""):
            paper_content = self.paper_content_extractor.extract_paper_content(
                publication=publication,
                context_size=graph.config.extraction_context_size,
                chunk_repetitions=graph.config.extraction_chunk_repetitions,
            )

        if paper_content is None:
            logger.warning((
                f"Skipping extraction for publication {publication.doi}. "
                "Because it propably has no full text or the extraction "
                "LLM config is missing."))
            return

        # add it to the graph
        self._add_paper_content_to_publication(
            graph, pub_uri, paper_content)

    def _add_paper_content_to_publication(self,
                                          graph: LocalKnowledgeGraph,
                                          pub_uri: URIRef,
                                          paper_content: PaperContent):
        """
        The following method goes through the content of the paper and
        adds them to the graph.

        Args:
            graph: The graph to build the block in.
            pub_uri: The URI of the publication node.
            paper_content: The content extracted from the paper.
        """
        if paper_content is None:
            logger.warning((
                f"Skipping adding paper content to publication {self.current_publication.doi}. "
                "Because the paper content is missing."))
            return

        self._add_problems(graph, pub_uri, paper_content)
        self._add_background_concepts(graph, pub_uri, paper_content)
        self._add_research_questions(graph, pub_uri, paper_content)
        self._add_contributions(graph, pub_uri, paper_content)

    def _add_problems(self,
                      graph: LocalKnowledgeGraph,
                      pub_uri: URIRef,
                      paper_content: PaperContent):
        """
        The method adds the research problems to the graph.

        Args:
            graph: The graph to build the block in.
            pub_uri: The URI of the publication node.
            paper_content: The content extracted from the paper.
        """
        if paper_content.research_problems and len(paper_content.research_problems) > 0:
            for context in paper_content.research_problems:
                graph.graph.add(
                    (pub_uri, EX.addressesResearchProblem, Literal(context.text)))

                self._add_tracing_for_literal(
                    graph, pub_uri, context.text, context)

    def _add_research_questions(self,
                                graph: LocalKnowledgeGraph,
                                pub_uri: URIRef,
                                paper_content: PaperContent):
        """
        The method adds the research questions to the graph.

        Args:
            graph: The graph to build the block in.
            pub_uri: The URI of the publication node.
            paper_content: The content extracted from the paper.
        """
        if paper_content.research_questions and len(paper_content.research_questions) > 0:
            for context in paper_content.research_questions:
                graph.graph.add(
                    (pub_uri, EX.hasResearchQuestion, Literal(context.text)))

                self._add_tracing_for_literal(
                    graph, pub_uri, context.text, context)

    def _add_contributions(self,
                           graph: LocalKnowledgeGraph,
                           pub_uri: URIRef,
                           paper_content: PaperContent):
        """
        The method adds the contributions to the graph.

        Args:
            graph: The graph to build the block in.
            pub_uri: The URI of the publication node.
            paper_content: The content extracted from the paper.
        """
        if paper_content.contributions and len(paper_content.contributions) > 0:
            for context in paper_content.contributions:
                # We create an new entity for each contribution
                concept_uri = RE[graph.get_new_id("contribution")]
                graph.graph.add((concept_uri, RDF.type, EX.Contribution))
                graph.graph.add(
                    (concept_uri, RDFS.label, Literal(context.name)))
                graph.graph.add((concept_uri, EX.hasDescription,
                                Literal(context.description)))
                graph.graph.add(
                    (pub_uri, EX.hasContribution, concept_uri))

                self._add_tracing_for_entity(
                    graph, concept_uri, context, pub_uri)

    def _add_background_concepts(self,
                                 graph: LocalKnowledgeGraph,
                                 pub_uri: URIRef,
                                 paper_content: PaperContent):
        """"
        The method adds the background concepts to the graph.

        Args:
            graph: The graph to build the block in.
            pub_uri: The URI of the publication node.
            paper_content: The content extracted from the paper.
        """
        if paper_content.background_concepts and len(paper_content.background_concepts) > 0:
            for context in paper_content.background_concepts:
                # We create an new entity for each concept
                concept_uri = RE[graph.get_new_id("backgroundConcept")]
                graph.graph.add((concept_uri, RDF.type, EX.BackgroundConcept))
                graph.graph.add(
                    (concept_uri, RDFS.label, Literal(context.name)))
                graph.graph.add((concept_uri, EX.hasDescription,
                                Literal(context.description)))
                graph.graph.add(
                    (pub_uri, EX.hasBackgroundConcept, concept_uri))

                self._add_tracing_for_entity(
                    graph, concept_uri, context, pub_uri)

    def _add_tracing_for_entity(self,
                                graph: LocalKnowledgeGraph,
                                entity_uid: str,
                                context: Union[Entity, TextWithOriginal],
                                parent_uid: str = None):
        """
        This methods adds a trace for the given entity in the graph to understand
        how a extraction was converted into a triple.

        Args:
            graph: The graph to build the block in.
            entity_uid: The UID of the entity to add the trace for.
            context: The context of the entity to add the trace for.
            parent_uid: The UID of the parent entity to add the trace for.
        """

        # Get the created triples
        relations = graph.get_relations_of_head_entity(
            knowledge=graph.get_entity_by_id(str(entity_uid))
        )
        if not relations:
            raise ValueError(f"Error adding tracing for entity {entity_uid}")

        for relation in relations:
            trace = ContextTrace(
                source_id=self.current_publication.doi,
                contexts=context.model_dump()
            )
            self.tracing.add_trace(str(relation), trace)
        if parent_uid:
            entity_text = graph.get_entity_by_id(str(entity_uid)).text
            self._add_tracing_for_literal(
                graph, parent_uid, entity_text, context)

    def _add_tracing_for_literal(self,
                                 graph: LocalKnowledgeGraph,
                                 parent_uid: str,
                                 literal: str,
                                 context: Union[Entity, TextWithOriginal]):
        """
        This methods adds a trace for the given literal in the graph to understand
        how a extraction was converted into a triple.

        Args:
            graph: The graph to build the block in.
            parent_uid: The UID of the parent entity to add the trace for.
            literal: The literal to add the trace for.
            context: The context of the entity to add the trace for.
        """
        relations = graph.get_relations_of_head_entity(
            knowledge=graph.get_entity_by_id(str(parent_uid))
        )
        added_triple = None
        for relation in relations:
            if relation.entity_object.text == literal:
                added_triple = relation
                break
        if not added_triple:
            raise ValueError(f"Error adding tracing for literal {literal}")
        if self.current_publication is None:
            raise ValueError("Current publication is not set.")
        tracing_context = ContextTrace(
            source_id=self.current_publication.doi,
            contexts=context.model_dump()
        )
        self.tracing.add_trace(str(added_triple), tracing_context)
