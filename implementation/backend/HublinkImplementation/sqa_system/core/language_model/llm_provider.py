# standard imports
import os

# local imports
from sqa_system.core.config.models.embedding_config import EmbeddingConfig
from sqa_system.core.data.secret_manager import SecretManager
from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.core.logging.logging import get_logger

# from .implementations.googleai_embedding_adapter import GoogleAIEmbeddingAdapter
from .implementations.ollama_llm_adapter import OllamaLLMAdapter
#from .implementations.huggingface_embedding_adapter import HuggingFaceEmbeddingAdapter
from .implementations.ollama_embedding_adapter import OllamaEmbeddingAdapter
from .implementations.openai_embedding_adapter import OpenAiEmbeddingAdapter
from .implementations.openai_llm_adapter import OpenAiLLMAdapter
#from .implementations.huggingfacepipeline_llm_adapter import HuggingFacePipelineLLMAdapter
# from .implementations.googleai_llm_adapter import GoogleAiLLMAdapter
from .implementations.vdl_llm_adapter import VDLLLMAdapter
from .implementations.vdl_embedding_adapter import VDLEmbeddingAdapter
from .enums.llm_enums import EndpointType, ValidationResult, EndpointEnvVariable
from .base.embedding_adapter import EmbeddingAdapter
from .base.llm_adapter import LLMAdapter

logger = get_logger(__name__)


class LLMProvider:
    """
    Provides instances of the LLMAdapter and EmbeddingAdapter based
    on the specified LLM configuration and embedding configuration.
    """

    def __init__(self):
        self.api_key_manager = SecretManager()

    def get_llm_adapter(self, llm_config: LLMConfig) -> LLMAdapter:
        """
        Returns an instance of the LLMAdapter for the specified LLM configuration.

        Args:
            llm_config (LLMConfig): The LLM configuration.
            
        Returns:
            LLMAdapter: An instance of the LLMAdapter based on the LLM configuration.
        """
        if llm_config.endpoint == EndpointType.OPENAI.value:
            self.validate_endpoint(EndpointType.OPENAI)
            llm_adapter = OpenAiLLMAdapter(llm_config)
            llm_adapter.prepare() 
            logger.debug("OpenAI LLM adapter prepared")
            return llm_adapter
        if llm_config.endpoint == EndpointType.HUGGINGFACE.value:
            self.validate_endpoint(EndpointType.HUGGINGFACE)
            llm_adapter = HuggingFacePipelineLLMAdapter(llm_config)
            llm_adapter.prepare()
            logger.debug("HuggingFacePipeline LLM adapter prepared")
            return llm_adapter
        if llm_config.endpoint == EndpointType.OLLAMA.value:
            llm_adapter = OllamaLLMAdapter(llm_config)
            llm_adapter.prepare()
            logger.debug("Ollama LLM adapter prepared")
            return llm_adapter
        if llm_config.endpoint == EndpointType.GOOGLEAI.value:
            self.validate_endpoint(EndpointType.GOOGLEAI)
            llm_adapter = GoogleAiLLMAdapter(llm_config)
            llm_adapter.prepare()
            logger.debug("GoogleAI LLM adapter prepared")
            return llm_adapter
        if llm_config.endpoint == EndpointType.VDL.value:
            self.validate_endpoint(EndpointType.VDL)
            llm_adapter = VDLLLMAdapter(llm_config)
            llm_adapter.prepare()
            logger.debug("VDL LLM adapter prepared")
            return llm_adapter
        return None

    def get_embeddings(self, embedding_config: EmbeddingConfig) -> EmbeddingAdapter:
        """
        Returns an instance of the EmbeddingAdapter for the specified embedding configuration.

        Args:
            embedding_config (EmbeddingConfig): The embedding configuration.
            
        Returns:
            EmbeddingAdapter: An instance of the EmbeddingAdapter based on the embedding configuration.
        """
        if embedding_config.endpoint == EndpointType.OPENAI.value:
            self.validate_endpoint(EndpointType.OPENAI)
            embedding_adapter = OpenAiEmbeddingAdapter(embedding_config)
            embedding_adapter.prepare()
            return embedding_adapter
        if embedding_config.endpoint == EndpointType.HUGGINGFACE.value:
            self.validate_endpoint(EndpointType.HUGGINGFACE)
            embedding_adapter = HuggingFaceEmbeddingAdapter(embedding_config)
            embedding_adapter.prepare()
            return embedding_adapter
        if embedding_config.endpoint == EndpointType.GOOGLEAI.value:
            self.validate_endpoint(EndpointType.GOOGLEAI)
            embedding_adapter = GoogleAIEmbeddingAdapter(embedding_config)
            embedding_adapter.prepare()
            return embedding_adapter
        if embedding_config.endpoint == EndpointType.OLLAMA.value:
            embedding_adapter = OllamaEmbeddingAdapter(embedding_config)
            embedding_adapter.prepare()
            return embedding_adapter
        if embedding_config.endpoint == EndpointType.VDL.value:
            self.validate_endpoint(EndpointType.VDL)
            embedding_adapter = VDLEmbeddingAdapter(embedding_config)
            embedding_adapter.prepare()
            return embedding_adapter

        raise ValueError(
            f"Unknown embedding endpoint: {embedding_config.endpoint}")

    def validate_endpoint(self, endpoint: EndpointType) -> ValidationResult:
        """
        Validates the specified endpoint by checking if the corresponding API key is present.

        Args:
            endpoint (EndpointType): The endpoint to be validated.
            
        Returns:
            ValidationResult: The result of the validation.
        """
        if endpoint == EndpointType.OPENAI:
            return self._validate_api_key(EndpointType.OPENAI)
        if endpoint == EndpointType.HUGGINGFACE:
            return self._validate_api_key(EndpointType.HUGGINGFACE)
        if endpoint == EndpointType.GOOGLEAI:
            return self._validate_api_key(EndpointType.GOOGLEAI)
        if endpoint == EndpointType.OLLAMA:
            return True
        if endpoint == EndpointType.VDL:
            return self._validate_api_key(EndpointType.VDL)
        return False

    def _validate_api_key(self, endpoint: EndpointType) -> ValidationResult:
        """
        Validates whether the API key for the specified endpoint is set in the environment variables.
        
        Args:
            endpoint (EndpointType): The endpoint to be validated.
        
        Returns:
            ValidationResult: The result of the validation.
        """
        try:
            api_key = self.api_key_manager.get_api_key(endpoint)
        except Exception:
            return ValidationResult.MISSING_API_KEY

        env_variable = EndpointEnvVariable.get_env_variable(endpoint)
        os.environ[env_variable.value] = api_key
        if not os.environ.get(env_variable.value):
            raise ValueError("Error setting environment variable")
        return ValidationResult.VALID

    def prepare_endpoint(self, endpoint: EndpointType, api_key: str):
        """
        Prepares the specified endpoint by saving the API key and 
        setting it in the environment variables.

        Args:
            endpoint (EndpointType): The endpoint to be prepared.
            api_key (str): The API key for the specified endpoint.
        """
        self.api_key_manager.save_api_key(endpoint, api_key)
        env_variable = EndpointEnvVariable.get_env_variable(endpoint)
        os.environ[env_variable.value] = api_key
