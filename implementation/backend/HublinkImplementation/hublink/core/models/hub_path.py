from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from core.data.models.triple import Triple


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
    dense_score: Optional[float] = Field(
        default=None,
        description=(
            "The dense semantic similarity score of the path for the given "
            "question."
        ))
    score: Optional[float] = Field(
        default=None,
        description=(
            "The final globally comparable path score used for selection and "
            "hub pruning. Equals dense_score for dense-only retrieval and the "
            "normalized path-level RRF score for hybrid retrieval."
        ))
    sparse_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Raw path scores keyed by sparse retrieval channel.")
    sparse_ranks: Dict[str, int] = Field(
        default_factory=dict,
        description="Zero-based global path ranks keyed by sparse channel.")
