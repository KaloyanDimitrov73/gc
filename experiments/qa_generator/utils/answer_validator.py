from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.language_model.prompt_provider import PromptProvider

from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class AnswerContextModel(BaseModel):
    """A model used for selecting relevant context for an answer."""
    is_answerable: bool = Field(
        default_factory=bool,
        description="Whether the question is answerable with the context")
    answer: Optional[str] = Field(
        default_factory=str,
        description="The answer to the question based on the chosen context.")
    contexts_for_answer: List[int] = Field(
        default_factory=list,
        description="The ids of the contexts used for generating the answer.")
    reasoning: str = Field(
        default_factory=str,
        description="The reasoning behind the answerability of the question")


class AnswerValidator:
    """
    The AnswerValidator class is responsible for validating whether a question can be answered
    with the given contexts. It uses a language model to validate the question and provide an
    answer if possible.

    Args:
        llm_adapter (LLMAdapter): An instance of LLMAdapter to interact with the language model.        
    """

    def __init__(self,
                 llm_adapter: LLMAdapter):
        self.llm_adapter = llm_adapter
        self.prompt_provider = PromptProvider()

    def validate_answer(self,
                        prompt_contexts_text: str,
                        question: str,
                        additional_context_info: Optional[str] = None) -> AnswerContextModel:
        """
        Prompts an LLM to validate whether a question can be answered with the given contexts.
        If the question can be answered, the LLM will also provide the answer with the reasoning 
        and the contexts used for the answer.

        Args:
            prompt_contexts_text (str): A text containing the contexts with their
                corresponding ids.
            question (str): The question to validate.
            additional_context_info (str, optional): Additional context information
                that might be relevant for the answer. Defaults to None.
        Returns:
            AnswerContextModel: A model containing the answer, the reasoning, and
                the contexts used for the answer.
        """

        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "qa_generation/answer_context_selection_prompt.yaml")
        llm_runnable = self.llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")

        retry_count = 3
        response = None
        errors_from_last_call = []
        while retry_count > 0:
            if errors_from_last_call:
                prompt_text += ("In your last response you made the following errors, " +
                                f"make sure to correct them this time: {errors_from_last_call}")
            prompt = PromptTemplate(
                template="**Task Description** \n" + prompt_text,
                input_variables=["contexts", "question",
                                 "additional_context_info"],
            )
            chain = prompt | llm_runnable.with_structured_output(
                schema=AnswerContextModel)

            inputs = {
                "contexts": prompt_contexts_text,
                "question": question,
                "additional_context_info": "None" if not additional_context_info else additional_context_info
            }
            final_prompt = prompt.format_prompt(**inputs)
            logger.debug(
                f"Final prompt for answer context finding: {final_prompt}")
            try:
                response = chain.invoke(inputs)
                break
            except Exception as e:
                logger.debug(
                    f"The LLM was unable to find contexts for the answer: {e} Retrying ...")
                retry_count -= 1
                errors_from_last_call.append(
                    str(e).replace("{", "{{").replace("}", "}}"))
        if not response:
            logger.debug("Failed to find contexts for answer.")
            return None
        return response
