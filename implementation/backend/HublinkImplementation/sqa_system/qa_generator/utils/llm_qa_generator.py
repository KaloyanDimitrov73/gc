import json
from typing import List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from typing_extensions import Annotated
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic.functional_validators import AfterValidator
from pydantic import BaseModel, Field
import language_tool_python

from sqa_system.qa_generator.question_classifier import QuestionClassifier
from sqa_system.core.data.models.qa_pair import QAPair
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.knowledge_base.knowledge_graph.storage.utils.graph_converter import GraphConverter
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models.triple import Triple
from sqa_system.core.data.models.knowledge import Knowledge
from sqa_system.core.language_model.prompt_provider import PromptProvider
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.data.extraction.paper_content_extractor import (
    TextWithOriginal,
    Entity
)

from .related_text_extractor import RelatedTextExtractor
from .answer_validator import AnswerValidator, AnswerContextModel
logger = get_logger(__name__)

# The following data models are the inputs and outputs of the LLM.
# They are converted into a dictionary format and used with the LangChain
# library to generate the question and answer pairs.


class QAGenerationModel(BaseModel):
    """The model used for the generation of a question and answer pair."""
    question: Optional[str] = Field(
        default=None,
        description="The generated question.")
    answer: Optional[str] = Field(
        default=None,
        description="The answer to the generated question.")
    context_ids: Optional[List[int]] = Field(
        default=None,
        description=("The ids of the selected context "
                     "used for the question generation."))


class AnswerOutputModel(BaseModel):
    """The output structure of the request."""
    generated_qa: List[QAGenerationModel]


def validate_context_item(v):
    """
    Custom validator to handle Union[TextWithOriginal, Entity].
    It checks the instance type and returns it if valid.
    """
    for item in v:
        if not isinstance(item, (TextWithOriginal, Entity)):
            raise ValueError(f"Invalid context item: {item}")
    return v


class ContextMapping(BaseModel):
    """A data class that allows mapping the context ids back to the source."""
    context_id: int = Field(..., description="The unique id of the context.")
    context: Annotated[List[Union[TextWithOriginal, Entity]],
                       AfterValidator(validate_context_item)] = Field(
        ...,
        description="The text of the context.")
    context_from_trace: bool = Field(
        ...,
        description="Indicates whether the context was traced back from the source document."
    )
    triples: Optional[List[Triple]] = Field(
        default=None,
        description="The triples of the context.")
    source_id: Optional[str] = Field(
        default=None,
        description="A ID tracing back the context to the source document.")
    source_name: Optional[str] = Field(
        default=None,
        description="The name of the source document.")


@dataclass
class QAGenerationData:
    """
    This data model is used for the Generator class and not the LLM.
    It defines those values that are needed to generate a question and answer pair.

    Args:
        context_text (str): A string that contains all the context from which the LLM
            should generate the question and answer pair.
        template_text (str): A string that contains the template that should be used
            to generate the question and answer pair.
        context_mapping (List[ContextMapping]): A list of ContextMapping objects that
            map the context ids back to the source.
        strategy_name (str): The name of the strategy that was used to generate the
            question and answer pair.
        validate_context (bool): A boolean that indicates whether the context should
            be validated or not. Default is True.
        topic_entity (Optional[Knowledge]): An optional Knowledge object that contains
            the topic entity from which the Hop Count should be calculated from.
        additional_requirements (Optional[str]): An optional string that contains
            additional requirements which are sent to the LLm to further instruct
            it on how to generate the question and answer pair.
    """
    context_text: str
    template_text: str
    context_mapping: List[ContextMapping]
    strategy_name: str
    validate_context: bool = True
    topic_entity: Optional[Knowledge] = None
    additional_requirements: Optional[str] = None


class LLMQAGenerator:
    """
    QA Generator class that uses a language model to generate questions and answers
    based on a given context and template.

    Args:
        graph (KnowledgeGraph): The knowledge graph that is used to generate the
            question and answer pair.
        llm_adapter (LLMAdapter): The language model adapter that is used to
            communicate with the language model.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 llm_adapter: LLMAdapter):
        self.graph = graph
        self.llm_adapter = llm_adapter
        self._prepare_utils()

    def _prepare_utils(self):
        self.question_classifier = QuestionClassifier(self.llm_adapter)
        self.prompt_provider = PromptProvider()
        self.graph_utils = GraphConverter(
            graph=self.graph, llm_adapter=self.llm_adapter)
        self.progress_handler = ProgressHandler()
        self.answer_validator = AnswerValidator(llm_adapter=self.llm_adapter)
        self.document_extractor = RelatedTextExtractor(
            llm_adapter=self.llm_adapter)

    def generate_golden_answer(self,
                               question: str,
                               golden_entities: List[str],
                               additional_information: str = "") -> str | None:
        """
        Tasks the LLM to generate a golden answer based on the given question and golden entities.

        Args:
            question (str): The question to generate the golden answer for.
            golden_entities (List[str]): The golden entities that should be included in the answer.
            additional_information (str): Additional information that should be included in the answer.
        """

        parser = StrOutputParser()
        prompt_provider = PromptProvider()
        prompt_text, _, _ = prompt_provider.get_prompt(
            "qa_generation/multi_fact_golden_answer_prompt.yaml")

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["question", "entities"],
            partial_variables={},
        )

        llm_runnable = self.llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        chain = prompt | llm_runnable | parser
        golden_entities_string = ", ".join(golden_entities)
        if additional_information:
            golden_entities_string += f" {additional_information}"
        response = chain.invoke(
            {"question": question, "entities": golden_entities_string})

        # Now we need to make sure that all the golden entities are included in the response
        for entity in golden_entities:
            if entity not in response:
                logger.error(
                    f"Golden entity {entity} is not included in the response: {response}"
                )
                return None
        return response

    def generate_qa_pairs(self, data: QAGenerationData) -> Tuple[List[QAPair], Set[str]]:
        """
        Runs the LLM to generate a question and answer pair based on the given context
        and template.

        Args:
            data (QAGenerationData): The data model that contains all the information
                needed to generate the question and answer pair.

        Returns:
            Tuple[List[QAPair], Set[str]]: A tuple containing a list of QAPair objects
                that contain the generated question and answer pair, and a set of
                golden predicates.
        """
        response = self._run_llm_for_generation(data)
        if not response:
            return [], set()

        qa_pairs: List[QAPair] = []
        golden_predicates = set()

        qa_task = self.progress_handler.add_task(
            string_id="qa_generation",
            description="Processing generated questions",
            total=len(response.generated_qa)
        )
        for generated_qa in response.generated_qa:
            logger.debug(f"Generated question: {generated_qa.question}")
            logger.debug(f"Initial answer: {generated_qa.answer}")
            logger.debug(f"Context ids: {generated_qa.context_ids}")

            # Check if a question was generated
            if not generated_qa.question:
                logger.debug("Question is invalid.")
                self.progress_handler.update_task_by_string_id(
                    qa_task, advance=1)
                continue
            try:
                validation_result = self._validate_context(
                    data, generated_qa)
                if validation_result is None:
                    self.progress_handler.update_task_by_string_id(
                        qa_task, advance=1)
                    continue

                metadata = validation_result["metadata"]
                filtered_triples = validation_result["filtered_triples"]
                generated_answer = validation_result["generated_answer"]

                new_qa_pair = QAPair(
                    question=generated_qa.question,
                    golden_answer=generated_answer,
                    golden_doc_chunks=metadata['golden_text_chunks'] if len(
                        metadata['golden_text_chunks']) > 0 else None,
                    source_ids=metadata['source_ids'],
                    topic_entity_id=data.topic_entity.uid if data.topic_entity else None,
                    topic_entity_value=data.topic_entity.text if data.topic_entity else None,
                    golden_triples=Triple.convert_list_to_string(
                        filtered_triples['golden_triples']),
                    is_generated_with=data.strategy_name,
                    based_on_template=data.template_text
                )

                qa_pairs.append(new_qa_pair)
                self.progress_handler.update_task_by_string_id(
                    qa_task, advance=1)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response: {e}")
                self.progress_handler.update_task_by_string_id(
                    qa_task, advance=1)
                continue
        self.progress_handler.finish_by_string_id(qa_task)
        return qa_pairs, golden_predicates

    def _validate_context(self,
                          data: QAGenerationData,
                          generated_qa: QAGenerationModel) -> dict | None:
        """
        Validates the generated question and answer pair by checking the context
        and the generated answer. It also checks if the generated question
        contains any golden entities.

        Args:
            data (QAGenerationData): The data model that contains all the information
                needed to generate the question and answer pair.
            generated_qa (QAGenerationModel): The generated question and answer pair.

        Returns:
            dict | None: A dictionary containing the metadata, filtered triples,
                and generated answer if the validation is successful, otherwise None.
        """
        # First we check whether the ids that the LLM provided are
        # actually in the context mapping
        if not self._check_if_ids_are_in_context_mapping(
                generated_qa, data.context_mapping):
            return None

        if data.validate_context:
            filtered_contexts, generated_answer = self._verify_contexts_for_answer(
                generated_qa, data.context_mapping, data.additional_requirements)
            if len(filtered_contexts) == 0:
                return None
        else:
            filtered_contexts = []
            for ctx_id in generated_qa.context_ids:
                for ctx in data.context_mapping:
                    if ctx.context_id == ctx_id:
                        filtered_contexts.append(ctx)
                        break
            generated_answer = generated_qa.answer

        # Get the metadata from the filtered contexts
        metadata = self._get_metadata(filtered_contexts)
        logger.debug(f"Golden triples: {metadata['golden_triples']}")
        logger.debug(
            f"Golden text chunks: {metadata['golden_text_chunks']}")

        if data.validate_context:
            filtered_triples = self._check_if_answer_can_be_infered_from_goldentriples(
                generated_qa,
                metadata,
                data.additional_requirements)
            if len(filtered_triples) == 0:
                return None
        else:
            filtered_triples = {
                "golden_triples": metadata["golden_triples"],
                "golden_source_ids": metadata["source_ids"]
            }

        return {
            "metadata": metadata,
            "filtered_triples": filtered_triples,
            "generated_answer": generated_answer
        }

    def _validate_answer(self,
                         generated_qa: QAGenerationModel):
        """
        Function that applies grammar correction using language_tool_python
        and then uses the LLM to correct the question.

        Args:
            generated_qa (QAGenerationModel): The generated question and answer pair.
        """
        # Sometimes the generated question has grammatical issues.
        # Here we try to correct them
        generated_qa.question = self._apply_grammar_correction(
            generated_qa.question)

        # Next we try to improve the question with another LLM call
        generated_qa.question = self._correct_question(
            generated_qa.question)

    def _get_metadata(self,
                      filtered_contexts: List[ContextMapping]) -> dict:
        """
        Extracts the source ids, source names, and golden ground truth from the
        contexts.

        Args:
            filtered_contexts (List[ContextMapping]): The filtered contexts that
                contain the relevant information for the metadata.

        Returns:
            A dictionary containing the source ids, source names, golden triples,
            golden predicates, and golden text chunks.
        """
        source_ids: List[str] = []
        source_names: List[str] = []
        golden_triples: List[Triple] = []
        golden_text_chunks: List[str] = []
        golden_predicates: Set[str] = set()

        # Define predicate priority
        predicate_priority = ["description", "example.org", "label"]

        for context_mapping in filtered_contexts:
            # Extract available predicates
            available_predicates = [
                triple.predicate for triple in context_mapping.triples]
            logger.debug(f"Available predicates: {available_predicates}")

            # Select the highest priority predicate available
            selected_predicate = ""
            for priority in predicate_priority:
                for predicate in available_predicates:
                    if priority.lower() in predicate.lower():
                        selected_predicate = predicate
                        logger.debug(("Priority selected predicate: "
                                     f"{selected_predicate}"))
                        break
                if selected_predicate:
                    break

            if not selected_predicate:
                # Append default metadata when no priority predicate is selected
                source_ids.append(context_mapping.source_id)
                source_names.append(context_mapping.source_name or "None")
                golden_triples.append(context_mapping.triples[0])

                if context_mapping.context_from_trace and context_mapping.context:
                    golden_text_chunks.append(
                        context_mapping.context[0].original_text)

                golden_predicates.add(context_mapping.triples[0].predicate)
                logger.debug(("Selected predicate: "
                             f"{context_mapping.triples[0].predicate}"))
                continue

            # Append metadata based on the selected priority predicate
            for idx, context_trace in enumerate(context_mapping.context):
                if context_mapping.triples[idx].predicate == selected_predicate:
                    source_ids.append(context_mapping.source_id)
                    source_names.append(context_mapping.source_name or "None")
                    golden_triples.append(context_mapping.triples[idx])

                    if context_mapping.context_from_trace:
                        golden_text_chunks.append(context_trace.original_text)

                    golden_predicates.add(selected_predicate)
                    break  # Exit the loop once the predicate is found

        # Convert the set of golden_predicates to a list for serialization
        return {
            "source_ids": source_ids,
            "source_names": source_names,
            "golden_triples": golden_triples,
            "golden_text_chunks": golden_text_chunks,
            "golden_predicates": list(golden_predicates)
        }

    def _check_if_answer_can_be_infered_from_goldentriples(
            self,
            generated_qa: QAGenerationModel,
            metadata: dict,
            additional_requirements: Optional[str] = None) -> dict:
        """
        Validates whether the answer is acutally based on the golden triples.

        Args:
            generated_qa (QAGenerationModel): The generated question and answer pair.
            metadata (dict): The metadata that contains the source ids, source names,
                golden triples, golden predicates, and golden text chunks.
            additional_requirements (Optional[str]): Additional requirements that
                should be included in the validation
        """
        prompt_contexts = []
        for k, triple in enumerate(metadata["golden_triples"]):
            prompt_contexts.append(
                (f"[Context ID: {k}, Context Data from Paper "
                 f"{metadata['source_names'][k]}: {str(triple)}]\n"))

        prompt_contexts_text = "\n".join(prompt_contexts)
        triple_answer_validation: AnswerContextModel = self.answer_validator.validate_answer(
            prompt_contexts_text=prompt_contexts_text,
            question=generated_qa.question,
            additional_context_info=additional_requirements
        )

        if not triple_answer_validation or not triple_answer_validation.is_answerable:
            logger.debug(
                ("Triple Validation: The generated question is not answerable."
                 f" Because: {triple_answer_validation.reasoning}"))
            return []
        golden_triples_filtered = []
        golden_source_ids_filtered = []
        should_continue = False
        for context_id in triple_answer_validation.contexts_for_answer:
            if context_id >= len(metadata["golden_triples"]):
                should_continue = True
                break
            golden_triples_filtered.append(
                metadata["golden_triples"][context_id])
            golden_source_ids_filtered.append(
                metadata["source_ids"][context_id])
        if should_continue:
            logger.debug(
                "Failed to find all contexts used for the answer.")
            return []
        logger.debug(
            f"Triple Validation Answer: {triple_answer_validation.answer}")
        return {
            "golden_triples": golden_triples_filtered,
            "golden_source_ids": golden_source_ids_filtered
        }

    def _check_if_ids_are_in_context_mapping(self,
                                             generated_qa: QAGenerationModel,
                                             context_mapping: List[ContextMapping]) -> bool:
        """
        Checks whether the ids that the LLM provided are actually in the context mapping.
        If not, the generated question is invalid and should be skipped.

        Args:
            generated_qa (QAGenerationModel): The generated question and answer pair.
            context_mapping (List[ContextMapping]): The context mapping that contains
                the context ids and the source ids.

        Returns:
            bool: True if the ids are in the context mapping, False otherwise.
        """
        contexts_valid = True
        if generated_qa.context_ids is None:
            logger.debug("No context ids were provided.")
            return False
        for index in generated_qa.context_ids:
            index_in_contexts = False
            for context in context_mapping:
                if context.context_id == index:
                    index_in_contexts = True
                    break
            if not index_in_contexts:
                logger.debug(f"Context with index {index} not found.")
                contexts_valid = False
                break
        return contexts_valid

    def _verify_contexts_for_answer(
            self,
            generated_qa: QAGenerationModel,
            context_mapping: List[ContextMapping],
            additional_requirements: Optional[str] = None) -> Tuple[List[ContextMapping], str]:
        """
        This method checks whether the generated question is actually answerable
        based on the context that was provided. 

        Args:
            generated_qa (QAGenerationModel): The generated question and answer pair.
            context_mapping (List[ContextMapping]): The context mapping that contains
                the context ids and the source ids.
            additional_requirements (Optional[str]): Additional requirements that
                should be included in the validation.

        Returns:
            Tuple[List[ContextMapping], str]: A tuple containing the filtered contexts
                and the generated answer.
        """
        prompt_contexts = []
        for context in context_mapping:
            # Only add the context if it is in the generated contexts
            if context.context_id not in generated_qa.context_ids:
                continue
            # Add the context to the prompt
            if context.source_name:
                prompt_contexts.append(
                    (f"[Context ID: {context.context_id}, Context from paper {context.source_name}: "
                     f"{context.context[0].original_text}]\n"))
            else:
                prompt_contexts.append(
                    f"[Context ID: {context.context_id}, Context Data: {context.context[0].original_text}]\n")

        prompt_contexts_text = "\n".join(prompt_contexts)
        original_text_answer_validation: AnswerContextModel = self.answer_validator.validate_answer(
            prompt_contexts_text=prompt_contexts_text,
            question=generated_qa.question,
            additional_context_info=additional_requirements
        )
        if not original_text_answer_validation or not original_text_answer_validation.is_answerable:
            logger.debug((
                "The generated question is not answerable. "
                f"Because: {original_text_answer_validation.reasoning}"))
            return [], ""

        # Only select the contexts that were used for the answer
        filtered_contexts: List[ContextMapping] = []
        for context_id in original_text_answer_validation.contexts_for_answer:
            contains_id = False
            for context in context_mapping:
                if context.context_id == context_id:
                    filtered_contexts.append(context)
                    contains_id = True
            if not contains_id:
                logger.debug(
                    "Failed to find all contexts used for the answer.")
                return [], ""

        return filtered_contexts, original_text_answer_validation.answer

    def _run_llm_for_generation(self, data: QAGenerationData) -> AnswerOutputModel:
        """
        This prompts the LLM for the generation of the question and answer pairs.

        Args:
            data (QAGenerationData): The data model that contains all the information
                needed to generate the question and answer pair.

        Returns:
            AnswerOutputModel: The output model that contains the generated question
                and answer pairs.
        """
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "qa_generation/qa_generation_prompt.yaml")

        retry_count = 3
        response = None
        errors_from_last_call = []
        while retry_count > 0:
            input_text = (
                "**Task Description** \n" + prompt_text)
            if errors_from_last_call:
                input_text += ("In your last response you made the following errors, " +
                               f"make sure to correct them this time: {errors_from_last_call}")
            prompt = PromptTemplate(
                template=input_text,
                input_variables=["contexts", "templates"],
            )
            chain = prompt | self.llm_adapter.llm.with_structured_output(
                schema=AnswerOutputModel)

            final_prompt = prompt.invoke({
                "contexts": data.context_text,
                "templates": data.template_text,
                "additional_requirements": (data.additional_requirements
                                            if data.additional_requirements else "None")
            })
            logger.debug(
                f"Final prompt for question generation: {final_prompt}")
            try:
                response = chain.invoke({
                    "contexts": data.context_text,
                    "templates": data.template_text,
                    "additional_requirements": (data.additional_requirements
                                                if data.additional_requirements else "None")
                })
                break
            except Exception as e:
                logger.debug(
                    f"The LLM was unable to generate a question: {e} Retrying ...")
                retry_count -= 1
                errors_from_last_call.append(
                    str(e).replace("{", "{{").replace("}", "}}"))
        if not response:
            logger.debug("Failed to generate a question.")
            return None
        return response

    def _apply_grammar_correction(self, question: str) -> str:
        """
        Applies the language_tool_python library to correct the grammar of the
        generated question.

        Args:
            question (str): The generated question to be corrected.

        Returns:
            str: The corrected question.
        """
        try:
            tool = language_tool_python.LanguageToolPublicAPI('en-US')
            matches = tool.check(question)
            question = language_tool_python.utils.correct(
                question, matches)
        except Exception as e:
            logger.error(
                f"Failed to correct the grammar of the generated question. Error: {e}")
        return question

    def _validate_golden_entities(self,
                                  question: str,
                                  golden_entities: List[str]) -> bool:
        """
        Validates whether the generated question contains any golden entities.

        Args:
            question (str): The generated question to be validated.
            golden_entities (List[str]): The list of golden entities that should
                not be included in the question.

        Returns:
            bool: True if the question does not contain any golden entities,
                False otherwise.
        """
        for golden_entity in golden_entities:
            if golden_entity.lower() in question.lower():
                logger.error(
                    "The LLM generated a question that contains a golden entity. Skipping..")
                logger.debug(f"Question: {question}")
                return False
        return True

    def _correct_question(self, question: str) -> str:
        """
        Receives a question and fixes the questions grammar and 
        sentence structure using an LLM.

        Args:
            question (str): The generated question to be corrected.

        Returns:
            str: The corrected question.
        """
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "qa_generation/question_correction_prompt.yaml")

        class JsonOutputObject(BaseModel):
            """The output structure of the request."""
            question: str

        parser = JsonOutputParser(pydantic_object=JsonOutputObject)

        prompt = PromptTemplate(
            template="\n{format_instructions}\n" + prompt_text,
            input_variables=["question"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )
        llm_runnable = self.llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        chain = prompt | llm_runnable | parser

        try:
            response = chain.invoke({"question": question})
        except Exception as e:
            logger.debug(f"There was an error correcting the question: {e}")
            return question
        if "question" in response:
            return response["question"]
        return question
