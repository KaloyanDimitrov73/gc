from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.config.models import RetrievalConfig


class Retriever(ABC):
    """
    Retriever class.
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = []

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    @classmethod
    @abstractmethod
    def create_config(cls,
                      retriever_type: str,
                      name: Optional[str] = None,
                      **kwargs) -> RetrievalConfig:
        """
        Creates a RetrievalConfig object with the specified parameters.
        It should be implemented by subclasses to create specific retrieval 
        configurations.
        
        Args:
            retriever_type (str): The type of the retriever.
            name (Optional[str]): The name of the retriever.
            **kwargs: Additional parameters for the configuration.
            
        Returns:
            RetrievalConfig: The created configuration object.
        """

    @classmethod
    def validate_config_params(cls, **kwargs):
        """
        Each retriever may have its own set of additional parameters associated with it
        that go beyond the base configuration. This method validates the additional parameters
        provided in the kwargs against the expected parameters defined in the class.
        
        Args:
            **kwargs: Additional parameters to validate.
        Raises:
            ValueError: If any of the additional parameters are missing or invalid.
        """
        additional_params = {}
        for param in cls.ADDITIONAL_CONFIG_PARAMS:
            is_in_kwargs = False
            # check if parameter is in kwargs
            for key, value in kwargs.items():
                if key == param.name:
                    is_in_kwargs = True
                    additional_params[key] = value
                    break
                if isinstance(key, AdditionalConfigParameter):
                    if key.name == param.name:
                        is_in_kwargs = True
                        additional_params[key.name] = value
                        break

            if not is_in_kwargs:
                raise ValueError(f"Parameter {param.name} is required.")
        return additional_params
