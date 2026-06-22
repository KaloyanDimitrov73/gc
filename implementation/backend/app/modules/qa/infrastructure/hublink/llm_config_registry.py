"""
Utility for loading and resolving LLMConfig entries from llm_configs.json.
"""
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional
import json
import logging

from backend.app.core.config import get_settings
from language_model.config.llm_config import LLMConfig

logger = logging.getLogger(__name__)


class LLMConfigRegistry:
    """
    Registry for model-name based LLMConfig lookup.
    """

    def __init__(self, config_file_path: Optional[Path] = None):
        settings = get_settings()
        self._config_file_path = config_file_path or Path(
            settings.retrieval_llm_configs_path
        )
        self._configs_by_model = self._load_configs_by_model()

    def _load_configs_by_model(self) -> Dict[str, LLMConfig]:
        if not self._config_file_path.exists():
            logger.warning("LLM config file not found at %s", self._config_file_path)
            return {}

        try:
            raw_configs = json.loads(self._config_file_path.read_text(encoding="utf-8"))
            configs_by_model: Dict[str, LLMConfig] = {}

            for raw_config in raw_configs:
                llm_config = LLMConfig.model_validate(raw_config)
                if llm_config.name_model in configs_by_model:
                    raise ValueError(
                        f"Duplicate llm model '{llm_config.name_model}' in {self._config_file_path}"
                    )
                configs_by_model[llm_config.name_model] = llm_config

            logger.info(
                "Loaded %d LLM configs from %s",
                len(configs_by_model),
                self._config_file_path
            )
            return configs_by_model
        except Exception as exc:
            logger.error(
                "Failed to load LLM configs from %s: %s",
                self._config_file_path,
                exc
            )
            return {}

    def get(self, llm_model: str) -> Optional[LLMConfig]:
        """
        Resolve a model name to a full LLMConfig.
        Returns a deepcopy to prevent accidental mutation of cached configs.
        """
        llm_config = self._configs_by_model.get(llm_model)
        return deepcopy(llm_config) if llm_config is not None else None

    def list_models(self) -> list[LLMConfig]:
        """Return all registered LLMConfig entries in insertion order."""
        return [deepcopy(cfg) for cfg in self._configs_by_model.values()]

