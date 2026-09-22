from abc import ABC
from typing import List, Type, TypeVar, Generic

from sqa_system.core.base.base_factory import BaseFactory

T = TypeVar('T', bound=BaseFactory)


class BaseFactoryRegistry(ABC, Generic[T]):
    """
    The base registry class that allows factories to 
    register themselves with the registry and other
    classes to retrieve the factories from the registry.
    """

    def __init__(self):
        self._factories = {}

    def register_factory(self, factory_type: str, factory: Type[T]):
        """
        Registers a factory with the given type.

        Args:
            factory_type (str): The type of the factory.
            factory (T): The factory to register.
        """
        self._factories[factory_type] = factory

    def get_factory_class(self, factory_type: str) -> type[T]:
        """
        Returns the factory class for the specified type.

        Args:
            factory_type (str): The type of the factory to retrieve.

        Returns:
            type[T]: The factory class for the specified type.

        Raises:
            ValueError: If no factory class is found for the specified type.
        """
        factory = self._factories.get(factory_type)
        if not factory:
            raise ValueError(f"Factory {factory_type} not found")
        return factory

    def get_factory_instance(self, factory_type: str) -> T:
        """
        Returns a factory instance for the specified type.

        Args:
            factory_type (str): The type of the factory to retrieve.

        Returns:
            T: The factory instance for the specified type.

        Raises:
            ValueError: If no factory instance is found for the specified type.
        """
        factory_class = self.get_factory_class(factory_type)
        factory = factory_class()
        return factory

    def get_all_factory_types(self) -> List[str]:
        """
        Returns a list of all factory types registered with the registry.

        Returns:
            List[str]: A list of all factory types registered with the registry.
        """
        return list(self._factories.keys())
