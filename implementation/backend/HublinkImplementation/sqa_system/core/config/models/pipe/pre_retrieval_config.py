from typing import Optional, Literal
from pydantic import Field

from sqa_system.core.config.models.pipe.pipe_config import PipeConfig
from sqa_system.core.config.models.llm_config import LLMConfig


class PreRetrievalConfig(PipeConfig):
    """Configuration for a pre retrieval processing pipe."""
    type: Literal["pre_retrieval_processing"] = "pre_retrieval_processing"
    pre_technique: str = Field(...,
                               description="The technique to be used for pre retrieval processing.")
    llm_config: Optional[LLMConfig]
    enabled: bool = Field(True,
                          description="Whether the pre retrieval processing is enabled or not.")

    def generate_name(self) -> str:
        if self.llm_config:
            return f"pre_retrieval_processing{self.pre_technique}_{self.llm_config.name}"
        return f"pre_retrieval_processing{self.pre_technique}"
