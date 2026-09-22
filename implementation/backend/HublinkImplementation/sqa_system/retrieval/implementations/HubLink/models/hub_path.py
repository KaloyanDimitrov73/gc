from typing import List, Optional
from pydantic import BaseModel, Field

from sqa_system.core.data.models.triple import Triple

class HubPath(BaseModel):
    """
    Represents a path inside of a 'Hub'. This path is a path from the
    root entity of the hub to either the end or the next hub.
    Also stores the textual description of the path and the embedding
    of the path text.
    """
    path_text: str = Field(...,
                           description="A textual description of the path.")
    path_hash: str = Field(...,
                           description="A hash of the path to uniquely identify it.")
    path: List[Triple] = Field(..., description="The triples of the path.")
    embedded_text: Optional[str] = Field(
        default=None, 
        description="The text that was embedded for the path.")
    score: Optional[float] = Field(
        default=None, 
        description="The score of the path based on the given question during retrieval.")
