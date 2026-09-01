import threading
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

    def retrieval(self, question: str, conversation_history: Optional[str] = None, cancel_event: Optional[threading.Event] = None) -> Optional[RetrievalAnswer]:
        """
        Retrieves the answer for a question using the strategy.

        Args:
            question (str): The question to retrieve the answer for.
            conversation_history: The last chat messages
            cancel_event (Optional[threading.Event]): Canceling of retrieval process
        """
        processed_question = self._process_question(question, conversation_history)
        if not processed_question:
            return None
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Canceling after: Retrieval")
            return None
        retrieval_answer = self._run_retrieval(processed_question, conversation_history, cancel_event)

        if retrieval_answer is not None:
            retrieval_answer.extracted_keywords = list(
                processed_question.keywords
            )
        return retrieval_answer

    @abstractmethod
    def _run_retrieval(self, processed_question: ProcessedQuestion, conversation_history: Optional[List[str]] = None, cancel_event: Optional[threading.Event] = None) -> Optional[RetrievalAnswer]:
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

    def _process_question(self, question: str, conversation_history: Optional[str] = None) -> ProcessedQuestion:
        """
        Prepares the question for retrieval.

        When ``extract_question_components`` is enabled, the LLM extracts
        semantic components and keywords. The components are embedded alongside
        the original question, while the keywords are stored in the returned
        processed-question model for downstream use. Keywords are not embedded.
        
        Args:
            question (str): The question to be processed.
            
        Returns:
            ProcessedQuestion: The processed question containing the original question,
                components, sparse routing keywords, and embeddings.
        """
        extracted_components: List[str] = []
        keywords: List[str] = []
        if self.settings.extract_question_components:
            extracted_components, keywords = (
                self._get_question_processing(question, conversation_history)
            )

        components = extracted_components

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
            keywords=keywords,
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

    def _get_question_processing(
            self, question: str, conversation_history: Optional[List[str]] = None) -> tuple[List[str], List[str]]:
        """
        Calls an LLM to extract question components and sparse routing keywords.
        
        Args:
            question (str): The question to process for retrieval.

        Returns:
            tuple[List[str], List[str]]: The extracted components and keywords.
        """
        prompt_provider = PromptProvider()
        prompt_text, _, _ = prompt_provider.get_prompt(
            "novel_retriever/question_processing_prompt.yaml")
        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["question", "chat_history"]
        )

        chain = prompt | self.llm_adapter.llm | parser
        response = chain.invoke(
            {
                "question": question,
                "chat_history": conversation_history or "",
            },
            config=_llm_call_config("hublink.question_processing"))
        logger.debug("Response from LLM for Question Processing: %s", response)

        components, keywords = self._extract_question_processing(response)

        logger.debug("Extracted question components: %s", components)
        logger.debug("Extracted sparse routing keywords: %s", keywords)

        return components, keywords

    @staticmethod
    def _extract_question_processing(
            llm_output: str) -> tuple[List[str], List[str]]:
        """
        Extracts the last valid processing dictionary from an LLM response.

        Some open-source models echo prompt examples or include explanatory
        text. Inspecting candidates from last to first allows the final answer
        to be parsed without requiring native structured-output support.
        
        Args:
            llm_output (str): The LLM output to parse.
            
        Returns:
            tuple[List[str], List[str]]: Normalized components and keywords.
                Invalid output safely produces two empty lists, which keeps
                retrieval dense-only.
        """
        dictionary_candidates = re.findall(
            r'\{[^{}]*\}', llm_output, flags=re.DOTALL
        )
        for candidate in reversed(dictionary_candidates):
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue

            if not isinstance(parsed, dict):
                continue
            if set(parsed) != {"components", "keywords"}:
                continue

            components = parsed["components"]
            keywords = parsed["keywords"]
            if not (BaseRetrievalStrategy._is_string_list(components)
                    and BaseRetrievalStrategy._is_string_list(keywords)):
                continue

            return (
                BaseRetrievalStrategy._normalize_string_list(components),
                BaseRetrievalStrategy._normalize_string_list(keywords),
            )

        logger.debug(
            "Question processing did not yield a valid dictionary.")
        return [], []

    @staticmethod
    def _is_string_list(value: object) -> bool:
        return (isinstance(value, list)
                and all(isinstance(item, str) for item in value))

    @staticmethod
    def _normalize_string_list(values: List[str]) -> List[str]:
        normalized: List[str] = []
        seen: set[str] = set()
        for value in values:
            stripped_value = value.strip()
            if not stripped_value or stripped_value in seen:
                continue
            normalized.append(stripped_value)
            seen.add(stripped_value)
        return normalized
