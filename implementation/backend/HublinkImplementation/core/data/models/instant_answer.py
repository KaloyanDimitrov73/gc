from typing import List

from pydantic import BaseModel, Field


class InstantAnswer(BaseModel):
    """
    Data object representing a directly generated answer without retrieval.

    Attributes:
        answer (str): The generated answer text.
        sources (List[str]): Sources referenced, if any (e.g. from conversation history).
    """
    answer: str = Field(..., description="The instantly generated answer.")
    sources: List[str] = Field(
        default_factory=list,
        description="Sources referenced in the answer, if applicable.")