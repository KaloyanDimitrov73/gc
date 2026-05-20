from abc import ABC, abstractmethod
from sqa_system.core.config.models.base.config import Config

class BaseFactory(ABC):
    """
    A base factory class that allows to create objects
    based on configurations.
    """

    @abstractmethod
    def create(self, config: Config, **kwargs):
        """
        Creates an object with the provided parameters.
        
        Args:
            config: The configuration object
            **kwargs: Additional parameters
        """
