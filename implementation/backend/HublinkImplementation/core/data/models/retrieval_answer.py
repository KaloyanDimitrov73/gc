from typing import List, Optional
from pydantic import BaseModel, Field

from .context import Context


class RetrievalAnswer(BaseModel):
    """
    A data object for an Answer from a retrieval process. 
    Each retriever of the SQA system returns a RetrievalAnswer object.
    """
    contexts: List[Context] = Field(
        ...,
        description="The context that was retrieved from the retriever.")
    retriever_answer: Optional[str] = Field(
        default=None,
        description="The answer generated based on the retrieval question which is optional.")
    sources: List[str] = Field(
        default_factory=list,
        description="Source identifiers (e.g. DOIs) of hubs actually referenced in the answer.")
