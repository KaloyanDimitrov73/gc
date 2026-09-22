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
from sqa_system.core.data.models import QAPair, Knowledge
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
class FromTopicEntityGeneratorOptions:
    """
    Options for the FromTopicEntityGenerator.

    Args:
        topic_entity_type (str, optional): The type that should be searched for when selecting a 
            topic entity.
        topic_entity_substring (str, optional): A substring that the topic entity must contain
            when selecting a topic entity.
        topic_entity (Knowledge, optional): If a topic entity is already known, it can be passed
            directly skipping the prior search.
        maximum_subgraph_size (int, optional): Restricts the size of the subgraph that is sent
            to the LLM. This is useful to avoid sending too much information to the LLM.
    """
    topic_entity_substring: Optional[str] = None
    topic_entity_type: Optional[str] = None
    topic_entity: Optional[Knowledge] = None
    maximum_subgraph_size: Optional[int] = 50


class FromTopicEntityGenerator(SubgraphStrategy):
    """
    A question answering strategy that generates questions and answers based on
    a knowledge graph. 

    This strategy generates questions and answers based on a topic entity
    and a publication. It first selects a topic entity from the knowledge graph
    and then generates a subgraph based on the topic entity. It then generates
    questions and answers based on the subgraph.

    Args:
        graph (KnowledgeGraph): The knowledge graph to be used for generation.
        llm_adapter (LLMAdapter): The language model adapter to be used for generation.
        from_topic_entity_options (FromTopicEntityGeneratorOptions): The options for the
            FromTopicEntityGenerator.
        options (GenerationOptions): The options for the generation.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 llm_adapter: LLMAdapter,
                 from_topic_entity_options: FromTopicEntityGeneratorOptions,
                 options: GenerationOptions):

        super().__init__(graph, llm_adapter, options)
        self.generator_options = from_topic_entity_options
        self.subgraph_filter = GraphPathFilter()
        self.single_fact_llm_qa_generator = FromSubgraphGenerator(
            graph=graph, llm_adapter=llm_adapter)
        self.question_classifier = QuestionClassifier(self.llm_adapter)
        self.predicate_filter_list = set()
        self.subgraph_builder = SubgraphBuilder(graph)

    @override
    def generate(self) -> List[QAPair]:
        """       
        Generates QA-Pairs based on the given template with a publisher as the topic
        entity.
        """

        # Get the topic entity
        if self.generator_options.topic_entity is not None:
            topic_entity = self.generator_options.topic_entity
        else:
            topic_entity_ids = self.graph.get_entity_ids_by_types(
                [self.generator_options.topic_entity_type])

            # Shuffle the ids
            random.shuffle(topic_entity_ids)

            topic_entity = None
            for topic_id in topic_entity_ids:
                topic_knowledge = self.graph.get_entity_by_id(topic_id)
                if self.generator_options.topic_entity_substring.lower() in topic_knowledge.text.lower():
                    topic_entity = topic_knowledge
                    break

        chosen_publication = self._select_publication_from_topic(topic_entity)

        if chosen_publication is None:
            logger.warning("No suitable publication found for generation.")
            return []

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

        # Prepare the additional requirements text
        requirements_text = None
        if self.options.additional_requirements:
            requirements_text = ""
            for i, req in enumerate(self.options.additional_requirements):
                requirements_text += f"{i + 1}. {req}\n"

        logger.debug(f"Generated subgraph with {len(subgraph)} triples.")
        qa_pairs, golden_predicates = self.single_fact_llm_qa_generator.generate_from_subgraph(
            SubgraphGeneratorOptions(
                root_entity=chosen_publication,
                subgraph=subgraph,
                template_text=self.options.template_text,
                strategy_name="publication_subgraph_strategy",
                additional_requirements=requirements_text,
                validate_contexts=self.options.validate_contexts,
                convert_path_to_text=self.options.convert_path_to_text,
                classify_qa_pairs=self.options.classify_questions,
                subgraph_size_limit=self.generator_options.maximum_subgraph_size
            )
        )

        # Finalize QA Pairs
        self.predicate_filter_list.update(golden_predicates)
        for qa_pair in qa_pairs:
            qa_pair.question = self.add_information_to_question(
                question=qa_pair.question,
                information=chosen_publication.text,
            )

        qa_pairs = self.enrich_qa_pairs(
            qa_pairs=qa_pairs,
            topic_entity=topic_entity,
            subgraph=subgraph)
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
            topic_entity=topic_entity
        )

        # shuffle the publications to get a random one
        random.shuffle(publications)
        return publications[0] if len(publications) > 0 else None
