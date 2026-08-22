from abc import ABC, abstractmethod
from dataclasses import dataclass
import ast
import re
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import numpy as np

from core.data.models import RetrievalAnswer, Triple
from core.progress.progress_handler import ProgressHandler
from hublink.core.models.hub import IsHubOptions, Hub
from hublink.core.models.hub_link_settings import HubLinkSettings
from hublink.core.models.hub_path import HubPath
from hublink.retrieval.models.processed_question import ProcessedQuestion
from hublink.retrieval.models.source_document_summary import SourceDocumentSummary
from language_model.base.embedding_adapter import EmbeddingAdapter
from language_model.base.llm_adapter import LLMAdapter
from language_model import PromptProvider
from knowledge_base.knowledge_graph.storage import KnowledgeGraph
from core.logging.logging import get_logger

from hublink.retrieval.utils.answer_generator import HubAnswer, AnswerGenerator
from hublink.retrieval.utils.hub_source_handler import HubSourceHandler
from hublink.core.hub_storage_manager import HubStorageManager

logger = get_logger(__name__)


def _llm_call_config(call_type: str) -> dict:
    """Build LangChain run config used by the shared LLM wrapper logger."""
    return {
        "metadata": {"llm_call_type": call_type},
        "tags": [f"llm_call_type:{call_type}"]
    }


@dataclass
class RetrievalStrategyData:
    """
    Data class for the retrieval strategy.

    Args: 
        graph (KnowledgeGraph): The knowledge graph.
        llm_adapter (LLMAdapter): The language model adapter.
        embedding_adapter (EmbeddingAdapter): The embedding adapter.
        settings (HubLinkSettings): The settings for the retrieval strategy.
        hub_storage_manager (HubStorageManager): The manager which manage the vector store for the retrieval.
        source_handler (HubSourceHandler, optional): The source handler
            for the retrieval that contains the linking data.
    """
    graph: KnowledgeGraph
    llm_adapter: LLMAdapter
    embedding_adapter: EmbeddingAdapter
    settings: HubLinkSettings
    hub_storage_manager: HubStorageManager
    source_handler: Optional[HubSourceHandler] = None


class BaseRetrievalStrategy(ABC):
    """
    A retrieval strategy for the HubLink retriever.

    Args:
        retrieval_data (RetrievalStrategyData): The data required 
            for the retrieval strategy.
    """

    def __init__(self,
                 retrieval_data: RetrievalStrategyData) -> None:
        self.graph = retrieval_data.graph
        self.llm_adapter = retrieval_data.llm_adapter
        self.embedding_model = retrieval_data.embedding_adapter
        self.settings = retrieval_data.settings
        self.hub_storage_manager = retrieval_data.hub_storage_manager
        self.hub_source_handler = retrieval_data.source_handler
        self._prepare_utils()

    def _prepare_utils(self):
        self.progress_handler = ProgressHandler()
        self.answer_generator = AnswerGenerator(
            graph=self.graph,
            llm=self.llm_adapter
        )

    def retrieval(self, question: str) -> Optional[RetrievalAnswer]:
        """
        Retrieves the answer for a question using the strategy.

        Args:
            question (str): The question to retrieve the answer for.
        """
        processed_question = self._process_question(question)
        if not processed_question:
            return None
        return self._run_retrieval(processed_question)

    @abstractmethod
    def _run_retrieval(self, processed_question: ProcessedQuestion) -> Optional[RetrievalAnswer]:
        """
        The main retrieval function that is called by the retrieval method.
        This function has to be implemented by the subclass.
        
        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the question, components, and embeddings.
        """
        

    def _get_hub_answers_directly(self, hub_scoring: List[Hub]) -> List[HubAnswer]:
        """
        Constructs HubAnswer objects from the hub paths without calling the LLM
        for per-hub partial answer generation.  The raw path descriptions are used
        directly as the hub_answer text so that the final answer generation LLM
        receives all path context in one step.

        Args:
            hub_scoring (List[Hub]): The list of hubs to build answers from.

        Returns:
            List[HubAnswer]: One HubAnswer per hub containing the concatenated
                path descriptions as hub_answer text.
        """

        hub_answers = []
        for hub in hub_scoring:
            if not hub.paths:
                continue
            context_texts = ""
            for index, path in enumerate(hub.paths):
                path_as_string = Triple.convert_list_to_string(path.path)
                context_texts += (
                    f"({index + 1}) **Description**: {path.path_text}; "
                    f"**Source**: {path_as_string}\n"
                )
            source_identifier = self.answer_generator._get_source_identifier_of_hub_entity(
                hub.root_entity
            )
            hub_answers.append(HubAnswer(
                source_identifier=source_identifier,
                source_name=hub.root_entity.entity.text,
                hub_answer=context_texts,
                relevant_paths=hub.paths,
                relevant_source_data=None,
            ))
        return hub_answers

    def _get_partial_answers(self,
                             processed_question: ProcessedQuestion,
                             hub_scoring: List[Hub]) -> List[HubAnswer]:
        """
        This function takes a list of hub scoring summaries and generates partial
        answers for each of them, meaning for each hub.
        This is done by looping over the summary objects and using it
        for the generation of the partial answer.

        This method is implemented with the possibility of parallelization
        using ThreadPoolExecutor if the number of workers is higher than 0.
        The reason we are not creating a thread pool with 1 worker is that we
        encountered a freezing issue on our server with the ThreadPoolExecutor
        when running on open source models. Therefore we have to enforce a
        single thread here.
        
        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the question, components, and embeddings.
            hub_scoring (List[Hub]): The list of hub objects that
                contains all necessary information to generate the partial answers.
        
        Returns:
            List[HubAnswer]: A list of partial answers for each hub scoring summary.            
        """
        hub_answers = []
        progress_task = self.progress_handler.add_task(
            description="Partial Answer Generation",
            total=len(hub_scoring),
            string_id="partial_answer_generation",
            reset=True
        )

        if self.settings.number_of_source_chunks > 1:
            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
                future_results = {}
                for scoring in hub_scoring:
                    future = executor.submit(
                        self._process_hub_scoring,
                        processed_question,
                        scoring)
                    future_results[future] = scoring

                for future in as_completed(future_results):
                    answer = future.result()
                    if answer:
                        hub_answers.append(answer)
                    self.progress_handler.update_task_by_string_id(
                        progress_task)
        else:
            for scoring in hub_scoring:
                answer = self._process_hub_scoring(
                    processed_question,
                    scoring)
                if answer:
                    hub_answers.append(answer)
        self.progress_handler.finish_by_string_id(progress_task)
        return hub_answers

    def _process_hub_scoring(self,
                             processed_question: ProcessedQuestion,
                             hub: Hub) -> Optional[HubAnswer]:
        """
        This is a helper function to parallelize the generation of partial answers
        for each hub.

        If a source handler is available, it links the hub to the source document
        and retrieves additional relevant information.
        
        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the question, components, and embeddings.
            hub (Hub): The hub object that contains
                all necessary information to generate the partial answer.
        
        Returns:
            Optional[HubAnswer]: A partial answer for the hub scoring summary if 
                the given context allows to create it.
        """
        return self.answer_generator.get_partial_answer_for_hub(
            hub_root_entity=hub.root_entity,
            question=processed_question.question,
            relevant_paths=hub.paths,
            source_document_data=self._get_link_data(
                processed_question=processed_question,
                hub=hub)
        )

    def _get_link_data(self,
                       processed_question: ProcessedQuestion,
                       hub: Hub) -> SourceDocumentSummary | None:
        """
        This function retrieves the linked data for a hub if the source database
        is available. It uses the hub source handler to get the relevant data
        and returns it.
        
        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the question, components, and embeddings.
            hub (Hub): The hub object that contains
                information to retrieve the linked data.
        
        Returns:
            SourceDocumentSummary | None: The linked data for the hub if available,
                otherwise None.
        """
        source_document_summary = None
        if self.hub_source_handler:
            source_document_summary = self.hub_source_handler.get_source_document_summary(
                processed_question=processed_question,
                hub_root_entity=hub.root_entity,
                n_results=self.settings.number_of_source_chunks
            )
        return source_document_summary

    def _prune_hubs(self,
                    hubs: List[Hub],
                    alpha: float = 5) -> List[Hub]:
        """
        Calculates the hub score as a weighted average of the hub paths scores,
        where the weights are computed using an exponential function of the score.

        Using this weighting, those scores that have a higher value are rated
        with a higher weigth than those that have lower scores. 

        It then prunes the hubs to the top k hubs based on the weighted hub score.
        
        Args:
            hub_scorings (List[Hub]): The list of hubs to be pruned.
            alpha (float): The scaling factor for the exponential function used
                to compute the weights. A higher value of alpha gives more weight
                to higher scores.
        
        Returns:
            List[Hub]: The pruned list of hubs sorted by their weighted hub score.
        """
        processed_hub_data: List[Hub] = []
        for hub_scoring in hubs:
            scores = [
                hub_path.score
                for hub_path in hub_scoring.paths
                if hub_path.score is not None
            ]
            if not scores:
                hub_scoring.hub_score = float("-inf")
                processed_hub_data.append(hub_scoring)
                continue
            # Compute weights using an exponential function of the scores
            weights = np.exp(alpha * np.array(scores))
            # Compute the weighted average score
            weighted_avg = np.sum(weights * np.array(scores)) / np.sum(weights)
            hub_scoring.hub_score = weighted_avg
            processed_hub_data.append(hub_scoring)

        # Sort the hubs by weighted hub score
        processed_hub_data.sort(key=lambda x: x.hub_score, reverse=True)

        # Prune to the top k hubs
        relevant_hubs = processed_hub_data[:self.settings.number_of_hubs]
        return relevant_hubs

    def _process_question(self, question: str) -> ProcessedQuestion:
        """
        Prepares the question for the retrieval process. If the option is enabled,
        it extracts the components of the question and embeds them.

        It also embeds the question itself.
        
        Args:
            question (str): The question to be processed.
            
        Returns:
            ProcessedQuestion: The processed question containing the original question,
                components, and embeddings.
        """
        components = []
        if self.settings.extract_question_components:
            components = self._get_question_components(question)

        logger.info("Embedding question")

        self.progress_handler.add_task(string_id="embedding_question", description="Embedding question", total=1, reset=True)
        all_texts = [question] + components
        embeddings = self.embedding_model.embed_batch(all_texts)
        if embeddings is None:
            raise RuntimeError("Embedding model returned None for question batch.")
        self.progress_handler.finish_by_string_id("embedding_question")

        return ProcessedQuestion(
            question=question,
            components=components,
            embeddings=embeddings
        )

    def _get_hub_paths_for_hub(self,
                               processed_question: ProcessedQuestion,
                               hub_id: str,
                               excluded_path_hashes: Optional[List[str]] = None
                               ) -> List[HubPath]:
        """
        Retrieves the top paths for a given hub ranked and scored by their relevance
        to the question.

        The storage search is responsible for returning unique logical paths.
        
        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the question, components, and embeddings.
            hub_id (str): The ID of the hub for which to retrieve the paths.
            excluded_path_hashes (List[str], optional): Paths that must not be
                returned by the storage search.
            
        Returns:
            List[HubPath]: The list of unique hub paths for the given hub.
        """
        return self.hub_storage_manager.similarity_search_by_hub_entity(
            query_embeddings=processed_question.embeddings,
            hub_entity_id=hub_id,
            n_results=self.settings.top_paths_to_keep,
            excluded_path_hashs=excluded_path_hashes,
        )

    def _get_question_components(self, question: str) -> List[str]:
        """
        This method calls an LLM to extract the components of the question.
        
        Args:
            question (str): The question where the components should be 
                extracted from.

        Returns:
            List[str]: The list of components extracted from the question.
        """
        prompt_provider = PromptProvider()
        prompt_text, _, _ = prompt_provider.get_prompt(
            "novel_retriever/question_processing_prompt.yaml")
        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["question"]
        )

        chain = prompt | self.llm_adapter.llm | parser
        response = chain.invoke(
            {"question": question},
            config=_llm_call_config("hublink.question_component_extraction"))
        logger.debug(f"Response from LLM for Question Components: {response}")

        question_components = self._extract_string_list(response)

        logger.debug(f"Extracted question components: {question_components}")

        return question_components

    def _extract_string_list(self, llm_output: str) -> List[str]:
        """
        This parser is used to extract a list of strings from the LLM output.
        We did not use JSON outputs here, because we want to test on open source
        models which are not able to return JSON outputs.

        This implementation however works generally well for all LLMs.
        
        Args:
            llm_output (str): The output from the LLM to extract the list from.
            
        Returns:
            List[str]: The list of strings extracted from the LLM output.
        """
        list_match = re.search(r'\[[^\]]*\]', llm_output)

        if list_match:
            list_str = list_match.group(0)
            try:
                # Attempt to safely evaluate the extracted string as a Python literal
                potential_list = ast.literal_eval(list_str)
                if (isinstance(potential_list, list) and
                        all(isinstance(item, str) for item in potential_list)):
                    return potential_list
            except (ValueError, SyntaxError):
                pass

            # Fallback: manually extract quoted strings within the brackets
            quoted_strings = re.findall(r'["\']([^"\']*)["\']', list_str)
            if quoted_strings:
                return quoted_strings

        # If no proper list found, try to extract quoted strings from the entire text
        quoted_strings = re.findall(r'["\']([^"\']*)["\']', llm_output)
        if quoted_strings:
            return quoted_strings

        # If no brackets found, check if the entire output is comma-separated items
        stripped_text = llm_output.strip()
        if ',' in stripped_text:
            return [item.strip() for item in stripped_text.split(',')]

        # If everything fails we return an empty list
        logger.debug(
            "Question component extraction did not yield a valid list.")
        return []
