from typing import List
from pydantic import BaseModel, Field


class ProcessedQuestion(BaseModel):
    """
    This data model represents the question that has been processed
    for retrieval. 
    """
    question: str = Field(...,
                            description="The original question.")
    components: List[str] = Field(default_factory=list,
                                  description="The components extracted from the question.")
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords indicating whether sparse retrieval is useful.")
    embeddings: List[List[float]] = Field(default_factory=list,
                                            description="The embeddings of the components and the question.")
