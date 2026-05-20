from sqa_system.core.config.models.base.config import Config

class EvaluatorConfig(Config):
    """Configuration class for Evaluators."""
    evaluator_type: str

    def generate_name(self) -> str:
        return f"{self.evaluator_type.lower()}"
