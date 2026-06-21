
from pathlib import Path
import json

from backend.app.core.config import get_settings
from sqa_system.retrieval.config.kg_retrieval_config import KGRetrievalConfig


class HublinkConfigLoader:
    """Loads the HubLink pipeline configuration and extracts the KG-retrieval pipe."""

    def __init__(self, config_file_path: Path | None = None):
        """
        Args:
            config_file_path: Path to the JSON pipeline config file. Falls back to
                ``settings.retrieval_kg_config_path`` when not provided.
        """
        settings = get_settings()
        self.default_config_file_path = config_file_path or Path(
            settings.retrieval_kg_config_path
        )

    def load_kg_retrieval_config(self) -> KGRetrievalConfig:
        """Parse the pipeline config file and return the KG-retrieval pipe config.

        Iterates over the ``pipes`` list in the JSON file and returns the first entry
        whose ``type`` equals ``"kg_retrieval"``.

        Returns:
            A validated ``KGRetrievalConfig`` instance.

        Raises:
            FileNotFoundError: If the config file does not exist at the resolved path.
            ValueError: If no pipe with ``type='kg_retrieval'`` is present in the file.
        """
        if not self.default_config_file_path.exists():
            raise FileNotFoundError(
                f"KG retrieval config file not found: {self.default_config_file_path}"
            )

        with open(self.default_config_file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        for pipe in config_data.get("pipes", []):
            if pipe.get("type") == "kg_retrieval":
                return KGRetrievalConfig(**pipe)

        raise ValueError("No pipe with type='kg_retrieval' found in JSON.")
