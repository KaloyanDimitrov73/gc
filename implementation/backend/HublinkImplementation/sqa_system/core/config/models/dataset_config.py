from typing import Optional
from typing_extensions import Annotated
from pydantic import Field
from sqa_system.core.config.models.base.config import Config


class DatasetConfig(Config):
    """Configuration for a dataset"""
    file_name: str
    loader: str
    loader_limit: Annotated[Optional[int], Field(ge=-1)]

    def generate_name(self) -> str:
        return (f"{self.file_name.replace(' ', '_').lower()}_"
                f"{self.loader.lower()}_limit{self.loader_limit}")
