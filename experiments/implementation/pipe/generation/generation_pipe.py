from pprint import pformat
from typing import List
from typing_extensions import override

from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from core.data.models import ContextType
from implementation.pipe.base.pipe import Pipe
from language_model.prompt_provider import PromptProvider
from implementation.config.models.pipe.generation_config import GenerationConfig
from language_model.llm_provider import LLMProvider
from implementation.models.pipe_io_data import PipeIOData
from implementation.core.logging import get_logger
logger = get_logger(__name__)


VECTOR_CONTEXT_EXPLANATION = """
The context is retrieved from a vector store. In this case:\n
- The information is divided into chunks.\n
- Chunks may be incomplete, cut off, or occasionally irrelevant to the question.\n
- Your task is to synthesize information from these chunks to answer the question.\n
"""

KG_CONTEXT_EXPLANATION = """
The context is retrieved from a knowledge graph. This means that:\n
- The context can be either an already generated answer, entities, or triples\n
- Triples are in the format: (entity, relation, entity).\n
- Your task is to use the context given to you to answer the question.\n
"""


class GenerationPipe(Pipe):
    """
    The generation pipe should be the last pipe in the pipeline.
    It is responsible for generating an answer based on the retrieved 
    context and the initial question.

    Args:
        config (GenerationConfig): The configuration for the generation pipe.
            It contains the LLM configuration and other parameters.
    """

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__()
        llm_provider = LLMProvider()
        self.llm = llm_provider.get_llm_adapter(config.llm_config)
        if self.llm is None:
            logger.error("The LLM could not be loaded")
            raise ValueError("The LLM could not be loaded")

    @override
    def _process(self, input_data: PipeIOData) -> PipeIOData:
        """
        The process method is responsible for generating an answer
        based on the retrieved context and the initial question.
        It uses the LLM to generate the answer and returns the updated
        input data with the generated answer.

        Args:
            input_data (PipeIOData): The input data that will be processed.
                It is extended or manipulated by the pipe.

        Returns:
            PipeIOData: The processed input data with the generated answer.
        """
        # Some retrievers already provide an generated answer. In that case
        # we can skip the answer generation here
        if input_data.generated_answer is not None:
            return input_data

        prompt_provider = PromptProvider()
        prompt_text, _, _ = prompt_provider.get_prompt(
            "pipes/answer_generation_prompt.yaml")

        # Prepare the context and explanation
        context_explanation = ""
        contexts = []
        for context in input_data.retrieved_context:
            if context_explanation == "":
                if context.context_type == ContextType.DOC:
                    context_explanation = VECTOR_CONTEXT_EXPLANATION
                elif context.context_type == ContextType.KG:
                    context_explanation = KG_CONTEXT_EXPLANATION
            contexts.append(str(context))
        if not contexts:
            input_data.generated_answer = "No context has been found for the given question."
            return input_data

        question = input_data.initial_question
        formatted_output = pformat(contexts, indent=2, width=100)
        logger.debug("Context: %s", formatted_output)

        response = self._get_llm_answer(context_explanation,
                                        contexts,
                                        question,
                                        prompt_text)
        input_data.generated_answer = str(response)

        return input_data

    def _get_llm_answer(self,
                        context_explanation: str,
                        contexts: List[str],
                        question: str,
                        prompt_text: str) -> str:
        """
        This method is responsible for generating the final answer
        using the LLM. It takes the context explanation, contexts,
        question, and prompt text as input and returns the generated
        answer as a string.

        Args:
            context_explanation (str): The explanation of the context.
            contexts (List[str]): The list of contexts to be used for generation.
            question (str): The question to be answered.
            prompt_text (str): The prompt text to be used for generation.

        Returns:
            str: The generated answer as a string.
        """
        llm_runnable = self.llm.llm
        if llm_runnable is None:
            raise ValueError("The LLM could not be loaded")

        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["context_explanation", "context", "question"],
        )

        final_prompt = prompt.invoke(
            {
                "context_explanation": context_explanation,
                "context": contexts,
                "question": question,
            }
        )
        # Log the prompt
        formatted_output = pformat(final_prompt, indent=2, width=100)
        logger.debug("Prompt: %s", formatted_output)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                chain = prompt | llm_runnable | parser
                response = chain.invoke(
                    {
                        "context_explanation": context_explanation,
                        "context": contexts,
                        "question": question,
                    }
                )
                break
            except Exception as e:
                logger.error(
                    "Failed to get final answer from generation "
                    f"pipe {attempt + 1} failed: {e}"
                )
                if attempt == max_retries - 1:
                    response = "Failed to generate an answer."
                    break
        return response
