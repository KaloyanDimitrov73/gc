from abc import ABC, abstractmethod
from typing import Any, Optional

import weave
from langchain_core.language_models import BaseChatModel

from language_model.config.llm_config import LLMConfig
from language_model.enums.llm_enums import ValidationResult
from core.logging.logging import get_logger

from .langchain_llm_wrapper import LangchainLLMWrapper

logger = get_logger(__name__)

class LLMAdapter(ABC):
    """
    A wrapper class for a Langchain LLM.
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.llm: Optional[BaseChatModel] = None

    def _set_llm(self, llm: BaseChatModel):
        self.llm = LangchainLLMWrapper(base_llm=llm)

    @weave.op()
    def generate(self, prompt: str) -> Any:
        """
        Generates an answer to the given prompt using the LLM.
        """
        if self.llm is None:
            raise ValueError("The LLM could not be loaded")
        answer = self.llm.invoke(prompt)
        return answer

    @abstractmethod
    def validate(self) -> ValidationResult:
        """
        Validates if the LLM is ready to be used. Needs to be implemented
        by the subclasses.

        Returns:
            LLMAdapterError: An enum indicating the result of the validation.
        """

    @abstractmethod
    def prepare(self):
        """
        A abstract method that needs to be implemented by the child class.
        It should prepare the LLM for use.
        """

    @staticmethod
    def llm_call_config(call_type: str) -> dict:
        """Build LangChain run config used by the shared LLM wrapper logger."""
        return {
            "metadata": {"llm_call_type": call_type},
            "tags": [f"llm_call_type:{call_type}"]
        }
