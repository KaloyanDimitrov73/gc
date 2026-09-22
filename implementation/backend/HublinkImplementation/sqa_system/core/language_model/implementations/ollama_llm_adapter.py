from typing_extensions import override
from langchain_ollama import ChatOllama
from sqa_system.core.language_model.enums.llm_enums import ValidationResult

from sqa_system.core.language_model.base.llm_adapter import LLMAdapter


class OllamaLLMAdapter(LLMAdapter):
    """Implementation of LLMAdapter for Ollama LLMs."""

    @override
    def prepare(self):

        max_tokens = self.llm_config.max_tokens
        if max_tokens == -1:
            max_tokens = None
        self._set_llm(ChatOllama(
            model=self.llm_config.name_model,
            temperature=self.llm_config.temperature,
            num_predict=self.llm_config.max_tokens
        ))

    @override
    def validate(self) -> ValidationResult:
        """
        Validates if the LLM is ready to be used.
        """
        return True
