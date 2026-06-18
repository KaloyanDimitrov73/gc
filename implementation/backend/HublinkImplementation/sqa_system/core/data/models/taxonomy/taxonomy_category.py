from typing import Dict
from pydantic import BaseModel, Field

from .taxonomy_type import TaxonomyType


class TaxonomyCategory(BaseModel):
    """
    Describes a category in a taxonomy.
    """
    description: str = Field(..., description="Description of the category.")
    types: Dict[str, TaxonomyType] = Field(
        ..., description="Dictionary of types under this category.")
