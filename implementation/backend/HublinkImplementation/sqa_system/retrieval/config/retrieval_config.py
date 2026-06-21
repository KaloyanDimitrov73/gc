from typing import Literal
from core import Config


class RetrievalConfig(Config):
    """Configuration for a retrieval pipe"""

    type: Literal["retrieval"]
    retriever_type: str
