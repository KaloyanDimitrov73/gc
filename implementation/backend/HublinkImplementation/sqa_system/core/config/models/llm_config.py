from typing import Optional
from typing_extensions import Annotated
from pydantic import Field

from sqa_system.core.config.models.base.config import Config


class LLMConfig(Config):
    """Configuration class for LLMs."""
    endpoint: str
    name_model: str
    temperature: Annotated[Optional[float | None], Field(default=0.0, ge=0.0, le=1.0)]
    max_tokens: Annotated[Optional[int], Field(default=-1, ge=-1)]
    reasoning_effort: Annotated[Optional[str], Field(default=None)]

    def generate_name(self) -> str:
        return f"{self.endpoint.lower()}_{self.name_model.lower()}_tmp{self.temperature}_maxt{self.max_tokens}_reasoning{self.reasoning_effort}"
