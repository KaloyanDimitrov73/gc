from enum import Enum


class EndpointType(Enum):
    """
    The following enum stores all the Endpoints that the SQASystem supports.

    It is used by the LLMProvider for determining which Endpoint to load
    given the config.
    """
    HUGGINGFACE = "HuggingFace"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"
    GOOGLEAI = "GoogleAI"
    VDL = "VDL"  # KIT Virtual Design Lab Server (OpenAI-compatible)
    KIT_TOOLBOX = "KitToolbox"  # KIT KI-Toolbox (OpenAI-compatible)

    @staticmethod
    def get_values() -> list:
        """Returns a list of all the values in the enum."""
        return [endpoint.value for endpoint in EndpointType]


class EndpointEnvVariable(Enum):
    """
    Each endpoint has a corresponding environment variable that stores the API key.
    Using this enum we map the endpoint to the environment variable.

    This allows other classes to store the API key in the environment variable
    before running the corresponding endpoint.
    """
    HUGGINGFACE_API_TOKEN = "HUGGINGFACEHUB_API_TOKEN"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    GOOGLE_API_KEY = "GOOGLE_API_KEY"
    VDL_API_KEY = "VDL_API_KEY"
    KIT_TOOLBOX_API_KEY = "KIT_TOOLBOX_API_KEY"

    @staticmethod
    def get_env_variable(endpoint_type: EndpointType) -> "EndpointEnvVariable":
        """
        Returns the environment variable for the specified LLM endpoint.

        Args:
            endpoint_type (EndpointType): The type of LLM endpoint.

        Returns:
            EndpointEnvVariable: The environment variable for the specified LLM endpoint.
        """
        if endpoint_type == EndpointType.HUGGINGFACE:
            return EndpointEnvVariable.HUGGINGFACE_API_TOKEN
        if endpoint_type == EndpointType.OPENAI:
            return EndpointEnvVariable.OPENAI_API_KEY
        if endpoint_type == EndpointType.GOOGLEAI:
            return EndpointEnvVariable.GOOGLE_API_KEY
        if endpoint_type == EndpointType.VDL:
            return EndpointEnvVariable.VDL_API_KEY
        if endpoint_type == EndpointType.KIT_TOOLBOX:
            return EndpointEnvVariable.KIT_TOOLBOX_API_KEY
        raise ValueError(f"Invalid endpoint type: {endpoint_type}")


class ValidationResult(Enum):
    """
    Enum for different validation results.
    This allows to check if the LLM is ready to be used.
    """
    VALID = "VALID"
    MISSING_API_KEY = "MISSING_API_KEY"
