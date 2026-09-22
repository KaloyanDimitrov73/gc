from typing import Optional, Literal
from pydantic import Field

from sqa_system.core.config.models.pipe.pipe_config import PipeConfig
from sqa_system.core.config.models.llm_config import LLMConfig


class PostRetrievalConfig(PipeConfig):
    """Configuration for a post retrieval processing pipe."""
    type: Literal["post_retrieval_processing"] = "post_retrieval_processing"
    post_technique: str = Field(...,
                                description="The technique to be used for post retrieval processing.")
    llm_config: Optional[LLMConfig]
    enabled: bool = Field(True,
                          description="Whether the post retrieval processing is enabled or not.")

    def generate_name(self) -> str:
        if self.llm_config:
            return f"post_retrieval_processing_{self.post_technique}_{self.llm_config.name}"
        return f"post_retrieval_processing_{self.post_technique}"
