import os
from abc import ABC, abstractmethod
from uuid import uuid4
from typing import Any, Dict, List
import json
import hashlib
from typing_extensions import override
from pydantic import BaseModel, Field, model_validator

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter


class Config(BaseModel, ABC):
    """Base class for configurations used in the SQA system."""
    uid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    name: str = Field(default="")
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        """
        Generates a hash based on the current state of the configuration.
        This property is cached after the first computation to improve performance.
        The cache is invalidated if any attribute of the Config instance changes.
        """
        return self._compute_hash()

    def _compute_hash(self) -> str:
        config_dict = self.to_dict(exclude_names=True)
        hash_value = hashlib.md5(json.dumps(
            config_dict, sort_keys=True).encode()).hexdigest()
        return hash_value

    def _update_config_mapping(self, hash_value: str):
        """
        This method adds a mapping with the config hash and the configuration into a 
        json file to allow to trace the configuration by hash.

        Args:
            hash_value (str): The hash value of the configuration.
        """
        fpm = FilePathManager()
        json_path = fpm.get_path("config_mapping.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4)
        with open(json_path, "r+", encoding="utf-8") as f:
            config_mapping = json.load(f)
            config_mapping[hash_value] = self.to_dict()
            f.seek(0)
            json.dump(config_mapping, f, indent=4)
            f.truncate()

    @abstractmethod
    def generate_name(self) -> str:
        """
        Generate a name based on the configs attributes
        This method should be implemented by subclasses.
        """

    @model_validator(mode="after")
    def set_name(self):
        """
        Validates the name attribute of the Config instance after 
        initialization.

        If the name attribute is empty, it generates a name based on 
        the configs attributes using the generate_name method.

        This is used for automatic name generation when the name is not provided.

        Returns:
            Config: The Config instance with the name attribute set.
        """
        if not self.name:
            self.name = self.generate_name()
        return self

    @classmethod
    def prepare_name_for_config(cls, name: str) -> str:
        """
        This function should be used if the name of the config should
        be user generated. This ensures that the config name is generated
        in such a way, that it can be easily processed in the future.

        Args:
            name (str): The name to prepare.

        Returns:
            str: The prepared name where all spaces are replaced with underscores
                and the class name is appended.
        """
        name = name.replace(" ", "_") + f"_{cls.__name__}"
        return name

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an instance of the class from a dictionary.

        Args:
            data: The dictionary containing the data to initialize the instance.

        Returns:
            An instance of the class with the data from the dictionary.
        """
        return cls(**data)

    @staticmethod
    def validate_config_params(additional_config_params: List[AdditionalConfigParameter],
                               **kwargs) -> Dict[str, Any]:
        """
        Validates that all required configuration parameters are present in the keyword arguments 
        and returns those parameters as a dictionary.
        Args:
            additional_config_params (List[AdditionalConfigParameter]): A list of required 
                configuration parameters.
            **kwargs: Arbitrary keyword arguments that may include the required parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the required parameter names as keys and 
                their values from the provided keyword arguments.
        """
        additional_params = {}
        for param in additional_config_params:
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

    def to_dict(self, exclude_names: bool = False) -> dict:
        """
        Converts the config object to a dictionary representation.

        Args:
            exclude_names (bool): If True, excludes the "name" key from all levels.

        Returns:
            dict: A dictionary representation of the object.
        """
        dict_repr = json.loads(self.model_dump_json())
        if exclude_names:
            dict_repr = self._remove_names_recursively(dict_repr)
        else:
            dict_repr = {"name": self.name, **dict_repr}
        return dict_repr

    def _remove_names_recursively(self, data: Any) -> Any:
        """
        Recursively removes "name" keys from dictionaries.

        Args:
            data (Any): The data structure to process.

        Returns:
            Any: The processed data with "name" keys removed.
        """
        if isinstance(data, dict):
            return {k: self._remove_names_recursively(v) for k, v in data.items() if k != "name"}
        if isinstance(data, list):
            return [self._remove_names_recursively(item) for item in data]
        return data

    def gather_all_configs(self, parent_string: str = "") -> Dict[str, "Config"]:
        """
        Returns a dictionary of all configurations in this config and its nested configs.
        The key is the name of the configuration and the value
        is the configuration object.

        Args:
            parent_string (str): The string to prepend to the keys.

        Returns:
            Dict[str, Config]: A dictionary of all configurations in this config and its nested configs.
        """
        configs: Dict[str, Config] = {self.name: self}

        for field_name, _ in self.model_fields.items():
            field_value = getattr(self, field_name)

            if isinstance(field_value, Config):
                configs[parent_string + field_name] = field_value
                configs.update(
                    field_value.gather_all_configs(field_name + "_"))
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, Config):
                        configs[parent_string + field_name] = item
                        configs.update(
                            item.gather_all_configs(field_name + "_"))
            elif isinstance(field_value, dict):
                for value in field_value.values():
                    if isinstance(value, Config):
                        configs[parent_string + field_name] = value
                        configs.update(
                            value.gather_all_configs(field_name + "_"))

        return configs

    @override
    def model_dump(self, *args, **kwargs):
        """
        Override model_dump from pydantic to include the name in the output.
        """
        dump = super().model_dump(*args, **kwargs)
        return {**{"name": self.name}, **dump}
