import hashlib
import json
from typing import Optional
from ..dataset_config import DatasetConfig
from ..base.config import Config
from ..llm_config import LLMConfig


class KnowledgeGraphConfig(Config):
    """
    Configuration class for a knowledge graph.
    """
    graph_type: str
    dataset_config: Optional[DatasetConfig] = None
    extraction_llm: Optional[LLMConfig] = None
    extraction_context_size: Optional[int] = 4000
    extraction_chunk_repetitions: Optional[int] = 2

    def _compute_hash(self) -> str:
        config_dict = self.to_dict(exclude_names=True)
        for key in ("force_cache_update", "force_publication_update"):
            config_dict.get("additional_params", {}).pop(key, None)
        return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()

    def generate_name(self) -> str:
        if self.dataset_config is None:
            return f"{self.graph_type.lower()}"
        return f"{self.graph_type.lower()}_{self.dataset_config.name}"
