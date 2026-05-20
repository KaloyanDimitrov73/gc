from abc import ABC
import json
import os
from typing import Dict, List, Type, TypeVar, Generic, Optional
from pydantic import ValidationError
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.config.models.base.config import Config

logger = get_logger(__name__)

T = TypeVar('T', bound=Config)


class ConfigurationManager(ABC, Generic[T]):
    """
    Base class for configuration managers.
    Implements the singleton pattern to ensure that only one instance of the class is created.
    """
    # Override in subclass
    DEFAULT_FILE_NAME: str = ""
    DEFAULT_ROOT_NAME: str = "configurations"
    CONFIG_CLASS: Type[T] = None

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.path_manager = FilePathManager()
            self.configs: Dict[str, T] = {}
            self.__class__._initialized = True

    def get_all_ids(self) -> List[str]:
        """
        Returns a list of all the config IDs in the configuration manager.

        Returns:
            List[str]: A list of all the IDs in the configuration manager.
        """
        return list(self.configs.keys())

    def get_all_configs(self) -> List[T]:
        """
        Returns a list of all configurations.

        Returns:
            List[T]: A list of all configurations.
        """
        return list(self.configs.values())

    def get_config_by_name(self, name: str) -> T:
        """
        Retrieve the configuration for the given name.

        Args:
            name (str): The name of the configuration.

        Returns:
            T: The configuration object.

        Raises:
            KeyError: If the configuration for the given name is not found.
        """
        for config in self.configs.values():
            if config.name == name:
                return config
        logger.error("Configuration for '%s' not found", name)
        raise KeyError(f"Configuration for '{name}' not found")

    def get_config(self, uid: str) -> T:
        """
        Retrieve the configuration for the given UID.

        Args:
            uid (str): The unique identifier of the configuration.

        Returns:
            T: The configuration object.

        Raises:
            KeyError: If the configuration for the given UID is not found.
        """
        if uid not in self.configs:
            logger.error("Configuration for '%s' not found", uid)
            raise KeyError(f"Configuration for '{uid}' not found")
        return self.configs[uid]

    def add_config(self, config: T):
        """
        Adds a configuration to the manager.

        Args:
            config (T): The configuration object to be added.
        """
        self.configs[config.uid] = config
        logger.info("Added configuration %s to manager", config.uid)

    def remove_config(self, uid: str):
        """
        Removes a config from the manager based on the given id.

        Args:
            uid (str): The unique identifier of the configuration to be removed.
        """
        logger.info("Removed configuration %s from manager")
        if uid in self.configs:
            del self.configs[uid]
            logger.info("Removed configuration %s from manager", uid)

    def save_configs_as_default(self):
        """
        Saves the configurations as the default settings to a file.
        """
        if not all([self.DEFAULT_FILE_NAME, self.CONFIG_CLASS]):
            raise NotImplementedError(
                "Subclass must define DEFAULT_FILE_NAME and CONFIG_CLASS")

        default_path = self.path_manager.get_path(self.DEFAULT_FILE_NAME)
        default_dir = os.path.dirname(default_path)
        if not os.path.exists(default_dir):
            os.makedirs(default_dir)

        config_dicts = [config.to_dict()
                        for config in self.configs.values()]
        with open(default_path, 'w', encoding='utf-8') as f:
            json.dump({self.DEFAULT_ROOT_NAME: config_dicts}, f, indent=4)
        logger.info("Saved default configurations to %s", default_path)

    def load_default_configs(self, only_if_current_list_empty:bool = True):
        """
        Loads the default configurations from the specified location of the 
        filepathmanager.
        
        Args:
            only_if_current_list_empty (bool): If True, loads the default
                configurations only if the current list of configs is empty.
        """
        if only_if_current_list_empty and self.configs:
            return
        if not all([self.DEFAULT_FILE_NAME, self.CONFIG_CLASS]):
            raise NotImplementedError(
                "Subclass must define DEFAULT_FILE_NAME and CONFIG_CLASS")
        self.load_configs_by_file_name(
            self.DEFAULT_FILE_NAME,
            self.DEFAULT_ROOT_NAME
        )

    def load_configs_by_file_name(self,
                                  file_name: str,
                                  root_name: str = "configurations"):
        """
        Loads configurations by a file name if the file is stored in the
        path_manager.

        Args:
            file_name (str): The name of the file to load the configurations from.
            root_name (str, optional): The main key in the
                JSON file that contains the configurations.
        """
        file_path = self.path_manager.get_path(file_name)
        if not file_path:
            logger.warning("%s file path is not valid", file_name)
            return
        
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("[]")
        
        self.load_configs_from_path(file_path, root_name)

    def load_configs_from_path(self,
                               file_path: str,
                               root_name: Optional[str] = None,
                               overwrite_existing: bool = False,
                               throw_on_error: bool = False):
        """
        Loads configurations from a given path.

        Args:
            file_path (str): The path to the file to load the configurations from.
            root_name (Optional[str]): The main key in the JSON file that contains the configurations.
            overwrite_existing (bool): If True, overwrites existing configurations.
            throw_on_error (bool): If True, raises an exception on error.
        """
        if not os.path.exists(file_path):
            logger.warning("Path %s not found", file_path)
            if throw_on_error:
                raise FileNotFoundError(f"Path {file_path} not found")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            logger.warning("%s file not found", file_path)
            if throw_on_error:
                raise FileNotFoundError(f"{file_path} file not found") from exc
            return
        except json.JSONDecodeError as exc:
            logger.error("%s file could not be parsed: %s", file_path, exc)
            if throw_on_error:
                raise json.JSONDecodeError(str(exc), exc.doc, exc.pos)
            return
        
        if overwrite_existing:
            self.configs = {}

        if root_name and root_name in data:
            config_list = data[root_name]
        elif isinstance(data, dict):
            config_list = [data]
        else:
            config_list = data

        for config_info in config_list:
            try:
                config = self.CONFIG_CLASS.from_dict(config_info)
                self.configs[config.uid] = config
            except ValidationError as e:
                if "name" in config_info:
                    logger.error(
                        "The configuration with name '%s' is not valid.", config_info["name"])
                logger.error("Skipping because of: %s", e)

        logger.info("Loaded %d configurations from %s",
                    len(self.configs), file_path)

    @classmethod
    def get_config_class(cls) -> Type[T]:
        """
        Returns the class of the configuration the manager is managing.

        Returns:
            type[Config]: The class of the configuration.
        """
        if not cls.CONFIG_CLASS:
            raise NotImplementedError("Subclass must define CONFIG_CLASS")
        return cls.CONFIG_CLASS
