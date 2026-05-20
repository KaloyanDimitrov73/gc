from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic
from sqa_system.core.config.models.base.config import Config

T = TypeVar('T')
C = TypeVar('C', bound=Config)


class BaseManager(ABC, Generic[T, C]):
    """
    A base manager class that is used to store and retrieve items from a dictionary.
    The dictionary is keyed by the config hash of the config object.
    Additionally it uses the singleton design pattern to ensure only one instance is created.
    """
    _instance = None
    _items: Dict[str, T] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BaseManager, cls).__new__(cls)
        return cls._instance

    @abstractmethod
    def _create_item(self, config: C) -> T:
        """
        Needs to be implemented by subclasses to create an item based on the config.
        """

    def get_item(self, config: C) -> T:
        """
        Retrieves an item from the manager based on the config hash of the config object.
        If the item is not found, it is created using the `_create_item` method.
        
        Args:
            config (C): The config object used to create or retrieve the item.
            
        Returns:
            T: The item associated with the config object.
        """
        if not isinstance(config, Config):
            raise TypeError(f"Expected Config object, got {type(config)}")

        store_key = config.config_hash

        if store_key not in self._items:
            self._items[store_key] = self._create_item(config)

        return self._items[store_key]
