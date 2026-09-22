from typing import List, Union, Annotated
from pydantic import Field

from sqa_system.core.config.models.base.config import Config
from .pipe.generation_config import GenerationConfig
from .retrieval.kg_retrieval_config import KGRetrievalConfig
from .retrieval.document_retrieval_config import DocumentRetrievalConfig
from .pipe.pre_retrieval_config import PreRetrievalConfig
from .pipe.post_retrieval_config import PostRetrievalConfig

PipeConfigs = Annotated[
    Union[GenerationConfig, 
          KGRetrievalConfig, 
          DocumentRetrievalConfig, 
          PreRetrievalConfig, 
          PostRetrievalConfig], Field(discriminator="type")]


class PipelineConfig(Config):
    """Configuration for a pipeline"""
    pipes: List[PipeConfigs] = Field(default_factory=list)

    def generate_name(self) -> str:
        content_hash = self.config_hash
        return f"pipeline_{content_hash}"
