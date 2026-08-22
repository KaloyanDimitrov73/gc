from typing_extensions import override

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from implementation.pipe.base.pipe import Pipe
from implementation.language_model import PromptProvider
from implementation.config.config_models.pipe.pre_retrieval_config import PreRetrievalConfig
from language_model.llm_provider import LLMProvider
from implementation.shared_models.pipe_io_data import PipeIOData
from implementation.core.logging import get_logger
logger = get_logger(__name__)


class QuestionAugmentationPipe(Pipe):
    """
    This pipe is responsible for augmenting the original question using the LLM.
    It has the task of enhancing the initial question to improve the retrieval process.

    Args:
        config (PreRetrievalConfig): The configuration for the question augmentation pipe.
            It contains the LLM configuration and other parameters.
    """

    def __init__(self, config: PreRetrievalConfig) -> None:
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
        The process method is responsible for augmenting the original question
        using the LLM. It uses the LLM to generate an augmented question
        and returns the updated input data with the augmented question.

        Args:
            input_data (PipeIOData): The input data that will be processed.

        Returns:
            PipeIOData: The processed input data with the augmented question.
        """
        if not self.enabled:
            return input_data

        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "pipes/question_augmentation_prompt.yaml")

        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["original_question"],
        )

        final_prompt = prompt.invoke(
            {"original_question": input_data.initial_question})
        logger.debug(f"Prompt for LLM: {final_prompt}")

        llm_runnable = self.llm.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        try:
            chain = prompt | llm_runnable | parser
            response = chain.invoke(
                {"original_question": input_data.initial_question})
        except Exception as e:
            logger.error(f"Failed to run the LLM: {e}")
            return input_data

        logger.debug(
            f"The original question is: {input_data.initial_question}")
        logger.debug(f"The augmented question is: {response}")

        input_data.retrieval_question = response
        return input_data
