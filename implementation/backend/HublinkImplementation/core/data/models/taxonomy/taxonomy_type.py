from typing import List, Optional
from pydantic import BaseModel, Field


class TaxonomyType(BaseModel):
    """
    Describes the type of a information entity in a taxonomy.
    """
    description: str = Field(...,
                             description="Description of the type.")
    examples: Optional[List[str]
                       ] = Field(default=None, description="An example of this type.")
