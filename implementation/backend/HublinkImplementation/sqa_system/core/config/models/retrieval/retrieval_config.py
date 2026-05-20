from typing import Literal
from sqa_system.core.config.models.pipe.pipe_config import PipeConfig


class RetrievalConfig(PipeConfig):
    """Configuration for a retrieval pipe"""

    type: Literal["retrieval"]
    retriever_type: str
