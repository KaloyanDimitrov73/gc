import random
from typing import List, Optional
from dataclasses import dataclass
from typing_extensions import override

from sqa_system.knowledge_base.knowledge_graph.storage.utils.subgraph_builder import (
    SubgraphBuilder,
    SubgraphOptions
)
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models import QAPair, Knowledge, Triple
from sqa_system.core.logging.logging import get_logger
from sqa_system.qa_generator.question_classifier import QuestionClassifier
from sqa_system.knowledge_base.knowledge_graph.storage.utils.graph_path_filter import (
    GraphPathFilter
)

from ...base.subgraph_strategy import SubgraphStrategy
from ...base.kg_qa_generation_strategy import GenerationOptions
from ...utils.from_subgraph_generator import FromSubgraphGenerator, SubgraphGeneratorOptions

logger = get_logger(__name__)


@dataclass
class PaperComparisonGeneratorOptions:
    """
    Options for the PaperComparisonGenerator.

    Args:
        first_publication (Knowledge): The first publication to be compared.
        second_publication (Knowledge): The second publication to be compared.
        topic_entity (Knowledge, optional): The topic entity that has to be reachable
            from both publications.
    """
    first_publication: Knowledge
    second_publication: Knowledge
    topic_entity: Optional[Knowledge] = None


class PaperComparisonGenerator(SubgraphStrategy):
    """
    A question answering strategy that generates questions and answers based on
    a knowledge graph.
    This strategy generates questions and answers that require the comparison
    of two publications. 


    Args:
        graph (KnowledgeGraph): The knowledge graph to be used for generation.
        llm_adapter (LLMAdapter): The language model adapter to be used for generation.
        options (GenerationOptions): The options for the PaperComparisonGenerator.
        comparison_options (PaperComparisonGeneratorOptions): The options for the
            PaperComparisonGenerator.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 llm_adapter: LLMAdapter,
                 options: GenerationOptions,
                 comparison_options: PaperComparisonGeneratorOptions):

        super().__init__(graph, llm_adapter, options)
        self.subgraph_filter = GraphPathFilter()
        self.single_fact_llm_qa_generator = FromSubgraphGenerator(
            graph=graph, llm_adapter=llm_adapter)
        self.question_classifier = QuestionClassifier(self.llm_adapter)
        self.predicate_filter_list = set()
        self.subgraph_builder = SubgraphBuilder(graph)
        self.comparison_options = comparison_options

    @override
    def generate(self) -> List[QAPair]:
        """       
        Generates QA-Pairs based on the given template with a publisher as the topic
        entity.
        """

        publications_subgraph: List[Triple] = []
        chosen_publications = [
            self.comparison_options.first_publication,
            self.comparison_options.second_publication
        ]
        for chosen_publication in chosen_publications:
            logger.debug(
                f"Chosen publication: {chosen_publication.text}")

            logger.debug("Generating publication subgraph...")
            _, subgraph = self.subgraph_builder.get_subgraph(
                chosen_publication,
                options=SubgraphOptions(
                    hop_amount=10,
                    go_against_direction=True
                )
            )
            publications_subgraph.extend(subgraph.root)
        # Prepare the additional requirements text
        requirements_text = None
        if self.options.additional_requirements:
            requirements_text = ""
            for i, req in enumerate(self.options.additional_requirements):
                requirements_text += f"{i + 1}. {req}\n"

        logger.debug(
            f"Generated subgraph with {len(publications_subgraph)} triples.")
        qa_pairs, golden_predicates = self.single_fact_llm_qa_generator.generate_from_subgraph(
            SubgraphGeneratorOptions(
                root_entity=self.comparison_options.topic_entity,
                subgraph=publications_subgraph,
                template_text=self.options.template_text,
                strategy_name="publication_subgraph_strategy",
                additional_requirements=requirements_text,
                validate_contexts=self.options.validate_contexts,
                convert_path_to_text=self.options.convert_path_to_text,
                classify_qa_pairs=self.options.classify_questions
            )
        )

        # Finalize QA Pairs
        self.predicate_filter_list.update(golden_predicates)
        for qa_pair in qa_pairs:
            qa_pair.question = self.add_information_to_question(
                question=qa_pair.question,
                information=chosen_publication.text,
            )

        if self.comparison_options.topic_entity:
            qa_pairs = self.enrich_qa_pairs(
                qa_pairs=qa_pairs,
                topic_entity=self.comparison_options.topic_entity,
                subgraph=publications_subgraph)
        return qa_pairs

    def _select_publication_from_topic(self, topic_entity: Knowledge) -> Knowledge | None:
        """
        Selects a publication that is directly reached from the topic entity.
        The publication is selected randomly from the list of publications
        that are directly reachable from the topic entity.

        Args:
            topic_entity (Knowledge): The topic entity to select the publication from.
        Returns:
            Knowledge | None: The selected publication or None if no publication
                was found.
        """

        publications = self.get_closest_publications_from_topic(
            topic_entity=topic_entity)

        if not publications:
            return None

        # shuffle the publications to get a random one
        random.shuffle(publications)
        return publications[0] if len(publications) > 0 else None
