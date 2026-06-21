from typing import Literal
from config.model.pipe.pipe_config import PipeConfig
from language_model.config.llm_config import LLMConfig

class GenerationConfig(PipeConfig):
    """Configuration for a generation pipe."""
    type: Literal["generation"] = "generation"
    llm_config: LLMConfig

    def generate_name(self) -> str:
        return f"{self.type.lower()}_{self.llm_config.name}"
