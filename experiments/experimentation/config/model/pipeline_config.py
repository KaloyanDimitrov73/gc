from typing import List, Union, Annotated
from pydantic import Field

from core import Config
from config.model.pipe.generation_config import GenerationConfig
from sqa_system.retrieval.config.kg_retrieval_config import KGRetrievalConfig
from sqa_system.retrieval.config.document_retrieval_config import DocumentRetrievalConfig
from config.model.pipe.pre_retrieval_config import PreRetrievalConfig
from config.model.pipe.post_retrieval_config import PostRetrievalConfig

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
