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
    embeddings: List[List[float]] = Field(default_factory=list,
                                            description="The embeddings of the components and the question.")
