from typing import List
from typing_extensions import override
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from implementation.pipe.base.pipe import Pipe
from language_model.prompt_provider import PromptProvider
from implementation.config.models.pipe.post_retrieval_config import PostRetrievalConfig
from language_model.llm_provider import LLMProvider
from implementation.models.pipe_io_data import PipeIOData
from implementation.core.logging import get_logger
logger = get_logger(__name__)


class RankingModel(BaseModel):
    """The output model for the ranking task."""
    reranked_ids: List[int] = Field(...,
                                    description="The reranked IDs of the retrieved contexts.")


class ReRankingPipe(Pipe):
    """
    The reranking pipe is responsible for reranking the retrieved contexts
    based on the initial question and the retrieved contexts using a language model.

    Args:
        config (PostRetrievalConfig): The configuration for the reranking pipe.
    """

    def __init__(self, config: PostRetrievalConfig) -> None:
        super().__init__()
        llm_provider = LLMProvider()
        self.llm = llm_provider.get_llm_adapter(config.llm_config)
        self.prompt_provider = PromptProvider()
        self.enabled = config.enabled
        if self.llm is None:
            logger.error("The LLM could not be loaded")
            raise ValueError("The LLM could not be loaded")

    @override
    def _process(self, input_data: PipeIOData) -> PipeIOData:
        """
        The process method is responsible for reranking the retrieved contexts
        based on the initial question and the retrieved contexts using a language model.
        It uses the LLM to rerank the contexts and returns the updated
        input data with the reranked contexts.

        Args:
            input_data (PipeIOData): The input data that will be processed.

        Returns:
            PipeIOData: The processed input data with the reranked contexts.
        """
        if not self.enabled:
            return input_data

        if input_data.retrieved_context is None:
            logger.warning("No retrieved contexts found. Skipping reranking.")
            return input_data

        response = self._run_llm_reranking(input_data)
        if response is None:
            logger.error("Failed to rerank the contexts")
            return input_data

        # Now we map the reranked IDs to the contexts
        reranked_ids = response.reranked_ids
        retrieved_context = input_data.retrieved_context.copy()

        if not all(0 <= idx < len(retrieved_context) for idx in reranked_ids):
            logger.error("Received invalid context indices from LLM")
            return input_data

        # Here we add the reranked ids
        included_indices = set(reranked_ids)
        input_data.retrieved_context = []
        for reranked_context in reranked_ids:
            input_data.retrieved_context.append(
                retrieved_context[reranked_context])

        # Then append any contexts that were not included in the reranking
        for i, context in enumerate(retrieved_context):
            if i not in included_indices:
                input_data.retrieved_context.append(context)

        return input_data

    def _run_llm_reranking(self, input_data: PipeIOData) -> PipeIOData:
        """
        The method responsible for running the LLM to rerank the contexts based on the input data.

        Args:
            input_data (PipeIOData): The input data that will be processed.

        Returns:
            PipeIOData: The processed input data with the reranked contexts.
        """
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "pipes/context_reranking_prompt.yaml")

        # Now we prepare the mapping
        context_string = ""
        for i, context in enumerate(input_data.retrieved_context):
            context_string += f"{i}: {context.text}\n"

        if self.llm.llm is None:
            raise ValueError("The LLM could not be loaded")

        parser = PydanticOutputParser(pydantic_object=RankingModel)

        max_retries = 3
        errors = []
        for attempt in range(max_retries):
            try:
                if errors:
                    prompt_text += "\n\n In your last call you made the following errors. Make sure you fix them this time:" + \
                        "\n".join(errors)
                prompt = PromptTemplate(
                    template=("Format Instructions: \n {format_instructions} \n" +
                              "Task Description: " + prompt_text),
                    input_variables=["question", "contexts"],
                    partial_variables={
                        'format_instructions': parser.get_format_instructions()}
                )

                chain = prompt | self.llm.llm | parser
                response = chain.invoke(
                    {
                        "question": input_data.initial_question,
                        "contexts": context_string,
                    }
                )
                break
            except Exception as e:
                logger.error(
                    "Failed to get final answer from generation "
                    f"pipe {attempt + 1} failed: {e}"
                )
                errors.append(str(e).replace("{", "").replace("}", ""))
                if attempt == max_retries - 1:
                    return None
        return response
