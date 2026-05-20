from sqa_system.core.config.models.base.config import Config


class EmbeddingConfig(Config):
    """Configuration class for embeddings."""
    endpoint: str
    name_model: str

    def generate_name(self) -> str:
        return f"{self.endpoint.lower()}_{self.name_model.lower()}"
