import json
from typing import Type, List, Any


class ConfigManager:
    def __init__(self, config_type):
        self._config_type = config_type
        self._configs: List[Any] = []

    def load_configs_from_path(
        self,
        file_path: str,
        overwrite_existing: bool = True,
        throw_on_error: bool = False,
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        self._configs = [self._config_type.model_validate(item) for item in data]

    def get_all_configs(self) -> List[Any]:
        return list(self._configs)


class ConfigManagerFactory:
    @staticmethod
    def get_config_manager_by_type(config_type) -> ConfigManager:
        return ConfigManager(config_type)
