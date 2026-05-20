from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from sqa_system.core.language_model.prompt_provider import PromptProvider
from sqa_system.knowledge_base.knowledge_graph.storage.utils.path_builder import PathBuilder
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models import Triple, QAPair, Knowledge, Subgraph
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


@dataclass
class GenerationOptions:
    """
    Options for the KGQAGenerationStrategy.

    Args:
        additional_requirements (str): Additional requirements forwarded to
            the generation.
        template_text (str): The text based on which the LLM should generate
            the questions.
        validate_contexts (bool): Whether to validate the contexts that have 
            been chosen by the LLM.
        convert_path_to_text (bool): Whether to convert the triple paths to
            text which may improve the context for the LLM.
        classify_questions (bool): Whether to classify the questions based on
            the question taxonomy.
    """
    additional_requirements: List[str] = field(default_factory=list)
    template_text: Optional[str] = None
    validate_contexts: bool = True
    convert_path_to_text: bool = True
    classify_questions: bool = True


class KGQAGenerationStrategy(ABC):
    """
    A interface for a question answering strategy that generates
    question and answer pairs based on a given knowledge graph.

    Args:
        graph (KnowledgeGraph): The knowledge graph to use for generation.
        llm_adapter (LLMAdapter): The LLM adapter to use for generation.
        options (GenerationOptions): Options for the generation.
    """

    def __init__(self, graph: KnowledgeGraph, llm_adapter: LLMAdapter, options: GenerationOptions):
        self.graph = graph
        self.llm_adapter = llm_adapter
        self.prompt_provider = PromptProvider()
        self.options = options

    @abstractmethod
    def generate(self) -> List[QAPair]:
        """
        Generate QAPairs based on the strategy.
        """

    def fix_golden_entities(self, golden_entities: List[str]) -> Tuple[int, List[str]]:
        """
        Fixes golden entities if the LLM only partially included
        the text of the entity.

        Args:
            golden_entities: The golden entities to fix.
        Returns:
            Tuple[int, List[str]]: The maximum distance and the fixed golden entities.
        """
        final_golden_entities = []
        # Because there can be cases where the LLM selected multiple
        # entities as golden entities even though we only have one
        # entity, here it is merged into one entity.
        golden_values = list(golden_entities.values())
        if len(golden_values) > 1 and golden_values[0][0] == golden_values[1][0]:
            final_golden_entities = [golden_values[0][0]]
        else:
            final_golden_entities = [entity[0]
                                     for entity in golden_entities.values()]
        max_distance = max(entity[1] for entity in golden_entities.values())
        return max_distance, final_golden_entities

    def get_paths_from_topic_to_golden_entities(self,
                                                subgraph: List[Triple],
                                                topic_entity: Knowledge,
                                                golden_entities: List[str]) -> List[List[Triple]]:
        """
        Retrieves the relation paths to get from the topic entity to the golden entities.

        Args:
            subgraph: The subgraph to search in.
            topic_entity: The topic entity to start from.
            golden_entities: The golden entities to search for.

        Returns:
            List[List[Triple]]: The relation paths to the golden entities.
        """
        path_builder = PathBuilder(subgraph)
        all_paths = path_builder.build_all_paths(
            topic_entity, include_tails=True, include_against_direction=True)
        golden_entity_paths = []
        for path in all_paths:
            path_has_golden = False
            path_has_topic = False
            current_path = []
            for relation in path:
                current_path.append(relation)
                if relation.entity_object and relation.entity_object.text in golden_entities:
                    path_has_golden = True
                if ((relation.entity_subject and relation.entity_subject.uid == topic_entity.uid) or
                        (relation.entity_object and relation.entity_object.uid == topic_entity.uid)):
                    path_has_topic = True
                if path_has_golden and path_has_topic:
                    break
            if path_has_golden and path_has_topic:
                golden_entity_paths.append(current_path)
        # Remove any duplicate paths
        golden_entity_paths = [list(path) for path in set(
            tuple(path) for path in golden_entity_paths)]
        return golden_entity_paths

    def enrich_qa_pairs(self,
                        qa_pairs: List[QAPair],
                        subgraph: Subgraph,
                        topic_entity: Knowledge) -> List[QAPair]:
        """
        Enriches the QA pairs with the topic entity and the hop amount.

        Args:
            qa_pairs: List of QAPair objects to process
            topic_entity: The starting entity of the the search
            subgraph: Knowledge graph subgraph containing relevant triples

        Returns:
            List[QAPair]: Filtered and enriched QA pairs with added metadata.
                Invalid pairs (hop_amount <= 0) are excluded.
        """
        final_qa_pairs = []
        for index, qa in enumerate(qa_pairs):
            logger.debug(f"Processing QA-Pair {index + 1}/{len(qa_pairs)}")
            hop_amount = self.calculate_hop_amount(
                root_entity=topic_entity,
                subgraph=subgraph,
                golden_triples=qa.golden_triples
            )
            if hop_amount <= 0:
                logger.warning(
                    f"The LLM generated a question with invalid hop amount of "
                    f"{hop_amount} hops.")
                continue

            qa.hops = hop_amount
            qa.topic_entity_value = topic_entity.text
            qa.topic_entity_id = topic_entity.uid
            final_qa_pairs.append(qa)
        return final_qa_pairs

    def calculate_hop_amount(self,
                             subgraph: List[Triple],
                             root_entity: Knowledge,
                             golden_triples: List[Triple]) -> int:
        """
        Calculates the amount of hops needed from the topic entity 
        to the golden entities. If there are multiple golden entities
        the largest distance is chosen.

        Args:
            subgraph: The subgraph to search in.
            root_entity: The topic entity to start from.
            golden_triples: The golden triples to search for.
        Returns:
            int: The maximum distance to the golden entities.
        """

        path_builder = PathBuilder(subgraph)
        all_paths: List[List[Triple]] = path_builder.build_all_paths(
            current=root_entity,
            include_tails=True,
            include_against_direction=True
        )
        max_hop_amount = -1
        for golden_triple in golden_triples:

            for path in all_paths:
                last_root_index = None
                golden_index = None

                for index, relation in enumerate(path):
                    relation_str = str(relation)
                    if relation_str == str(golden_triple):
                        # We found the golden triple in this path
                        golden_index = index

                    # Update the last occurrence of root_entity
                    if root_entity.uid in {relation.entity_subject.uid, relation.entity_object.uid}:
                        last_root_index = index

                if golden_index is not None and last_root_index is not None:
                    # Ensure that the golden triple appears after the last root occurrence
                    if golden_index > last_root_index:
                        hop_amount = golden_index - last_root_index
                        max_hop_amount = max(max_hop_amount, hop_amount)
                    elif last_root_index == golden_index:
                        max_hop_amount = 0

        if max_hop_amount == -1:
            return -1
        return max_hop_amount + 1

    def add_information_to_question(self, information: str, question: str) -> str:
        """
        Uses an LLM to add additional information to a question.

        Args:
            information: The additional information to add to the question
            question: The question to add the information to

        Returns:
            str: The question with the additional information added
        """
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "qa_generation/add_information_to_question_prompt.yaml")

        prompt = PromptTemplate(
            template=prompt_text,
            variables={
                "additional_information": information,
                "question": question
            }
        )

        llm_runnable = self.llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        chain = prompt | llm_runnable | StrOutputParser()

        try:
            response = chain.invoke({"question": question,
                                     "additional_information": information})
        except Exception as e:
            logger.debug(
                f"There was an error adding information to the question: {e}")
            return question
        return response

    def get_closest_publications_from_topic(self, topic_entity: Knowledge, visited: set = None) -> List[Knowledge]:
        """
        Recursively retrieves publications considered "closest" to the given topic entity.

        The method first checks if the `topic_entity` itself is a publication allowed
        for generation. If not, it inspects its immediate neighbors in the graph.
        If no publications are found at this stage (i.e., at distance 0 or 1 from
        the topic_entity), the search continues recursively from these neighbors.

        Args:
            topic_entity: The entity from which to start the search for publications.
            visited: An optional set of entity UIDs that have already been
                visited during the search. This is used to avoid cycles and redundant
                processing. If `None`, a new set will be initialized for the current
                search. This set is modified by the function by adding the UID of
                the `topic_entity` being processed in each call.
        Returns:
            List[Knowledge]: A list of `Knowledge` objects representing the found
                publications. An empty list is returned if the `topic_entity`
                has already been visited (as per the `visited` set) or if no
                suitable publications are found through the search.
        """
        if visited is None:
            visited = set()
        if topic_entity.uid in visited:
            return []

        visited.add(topic_entity.uid)

        if self.graph.is_publication_allowed_for_generation(topic_entity):
            # The topic entity is a publication itself
            return [topic_entity]

        # Select publication from the topic
        tail_relations = self.graph.get_relations_of_tail_entity(
            topic_entity)
        head_relations = self.graph.get_relations_of_head_entity(
            topic_entity)

        publications = []
        for relation in tail_relations:
            if self.graph.is_publication_allowed_for_generation(relation.entity_subject):
                publications.append(relation.entity_subject)

        for relation in head_relations:
            if self.graph.is_publication_allowed_for_generation(relation.entity_object):
                publications.append(relation.entity_object)

        if len(publications) == 0:
            # Traverse deeper to find publications
            for relation in tail_relations:
                if self.graph.is_intermediate_id(relation.entity_object.uid):
                    publications.extend(self.get_closest_publications_from_topic(
                        relation.entity_subject, visited))
            for relation in head_relations:
                if self.graph.is_intermediate_id(relation.entity_subject.uid):
                    publications.extend(self.get_closest_publications_from_topic(
                        relation.entity_object, visited))
        return publications
