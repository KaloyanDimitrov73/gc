import os
import re
from typing import Any, Dict, Iterator, List, Optional
from typing_extensions import override
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    ChatResult
)

from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.language_model.enums.llm_enums import (
    ValidationResult,
    EndpointEnvVariable,
    EndpointType
)
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.language_model.errors.api_key_missing_error import APIKeyMissingError

logger = get_logger(__name__)


def remove_special_tokens_and_content(text: str, patterns: List[str]) -> str:
    """
    Removes special tokens that a model may include in its output.
    E.g. special tokens that indicate the start or end of a prompt.

    Args:
        text: The text to remove special tokens from.
        patterns: A list of regex patterns to match special tokens.
    """
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text.strip()


def remove_wrapping_backticks(text: str) -> str:
    """
    Removes triple backticks from the start and end of the text if they are present.
    Preserves any triple backticks that are part of the content.

    Args:
        text: The text to remove wrapping triple backticks from.
    """
    # Pattern to match starting triple backticks with optional language specifier
    start_pattern = r"^```(\w+)?\s*"
    # Pattern to match ending triple backticks
    end_pattern = r"\s*```$"

    # Remove starting triple backticks
    text = re.sub(start_pattern, "", text)
    # Remove ending triple backticks
    text = re.sub(end_pattern, "", text)

    return text.strip()


class CustomHuggingFaceWrapper(BaseChatModel):
    """
    A custom wrapper for Hugging Face models that removes special tokens and their 
    content from the output.

    Based on the implementation of `ChatHuggingFace` from `langchain_huggingface`
    and the wiki of langchain: https://python.langchain.com/docs/how_to/custom_chat_model/
    """

    llm: ChatHuggingFace
    special_patterns: Optional[List[str]] = None
    model_name: str

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response and remove special tokens and their content."""
        # Call the underlying LLMs _generate method
        # pylint: disable=protected-access
        response = self.llm._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )

        logger.debug(f"Response from Hugging Face model: {str(response)}")

        # Iterate over generations and remove special tokens and their content
        for generation in response.generations:
            original_content = generation.message.content
            # Remove special tokens and their content
            clean_content = remove_special_tokens_and_content(
                original_content, self.special_patterns)
            # Remove wrapping triple backticks
            clean_content = remove_wrapping_backticks(clean_content)
            generation.message.content = clean_content

        return response

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Call the underlying LLMs _generate method
        # We disable pylint here because the method is copied from the
        # above mentioned source implementation
        # pylint: disable=protected-access
        response = self.llm._generate(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )

        # Iterate over generations and remove special tokens and their content
        for generation in response.generations:
            original_content = generation.message.content
            # Remove special tokens and their content
            clean_content = remove_special_tokens_and_content(
                original_content, self.special_patterns)
            # Remove wrapping triple backticks
            clean_content = remove_wrapping_backticks(clean_content)
            generation.message.content = clean_content

        return response

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the response and remove special tokens and their content."""
        # Stream the response from the underlying LLM
        # We disable pylint here because the method is copied from the
        # above mentioned source implementation
        # pylint: disable=protected-access
        for chunk in self.llm._stream(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        ):
            if isinstance(chunk.message, AIMessageChunk):
                original_content = chunk.message.content
                clean_content = remove_special_tokens_and_content(
                    original_content, self.special_patterns)
                chunk.message.content = clean_content
            yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "custom-huggingface-wrapper"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
        }


class HuggingFacePipelineLLMAdapter(LLMAdapter):
    """
    Implementation of the LLMAdapter interface for Hugging Face LLMs.

    Uses a singleton pattern to ensure that only one instance of the
    adapter is created as subsequent calls would load the model again.

    Args:
        llm_config: The configuration for the LLM
    """

    _instances = {}
    _initialized = False

    def __new__(cls, llm_config: LLMConfig):
        config_key = llm_config.config_hash
        if config_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[config_key] = instance
        return cls._instances[config_key]

    def __init__(self, llm_config: LLMConfig):
        if not self._initialized:
            super().__init__(llm_config)
            self.llm = None
            self._model_is_loaded = False
            self.prepare()
            self._initialized = True

    @override
    def prepare(self):
        """Prepares the LLM for use"""
        validation_result = self.validate()
        if validation_result == ValidationResult.MISSING_API_KEY:
            raise APIKeyMissingError("Hugging Face API key is missing")

        if not self._model_is_loaded:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                llm = HuggingFacePipeline.from_model_id(
                    model_id=self.llm_config.name_model,
                    task="text-generation",
                    device_map="auto",
                    pipeline_kwargs={
                        "max_new_tokens": self.llm_config.max_tokens,
                        "trust_remote_code": True,
                    },
                    model_kwargs={"quantization_config": quantization_config},
                )

                # Based on: https://medium.com/@vishnuchirukandathramesh/how-to-run-mistral-7b-on-free-version-of-google-colab-e0effd9c6a12
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

                # Wrap the ChatHuggingFace with our CustomHuggingFaceWrapper
                chat_llm = ChatHuggingFace(
                    llm=llm,
                    model_kwargs={
                        "skip_special_tokens": True
                    }
                )

                special_patterns = [
                    r"(?s)<s>\[INST\].*?\[/INST\]",
                    r"(?i)</s>",
                ]

                self._set_llm(CustomHuggingFaceWrapper(
                    llm=chat_llm,
                    special_patterns=special_patterns,
                    model_name=self.llm_config.name_model,
                    cache=False
                ))

                self._model_is_loaded = True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Hugging Face model: {str(e)}") from e

    @override
    def validate(self) -> ValidationResult:
        # check if environment variable is set
        endpoint_var = EndpointEnvVariable.get_env_variable(
            EndpointType.HUGGINGFACE).value
        if endpoint_var not in os.environ:
            return ValidationResult.MISSING_API_KEY
        login(token=os.environ[endpoint_var])
        return ValidationResult.VALID
