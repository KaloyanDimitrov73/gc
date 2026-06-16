from typing import List, Optional
from pydantic import BaseModel, Field

from sqa_system.core.data.models.context import Context


class PipeIOData(BaseModel):
    """A object that represents the input of a pipe."""
    initial_question: str = Field(
        ...,
        description="The question that was inputted into the pipeline.")
    retrieval_question: str = Field(
        default=None,
        description="The actual question that was forwarded to the retriever.")
    topic_entity_id: Optional[str] = Field(
        default=None,
        description="An identifier of a knowledge entity that can be used as an entry point.")
    topic_entity_value: Optional[str] = Field(
        default=None,
        description="The value of the knowledge entity that can be used as an entry point.")
    retrieved_context: List[Context] = Field(
        default_factory=list,
        description="The context that was retrieved from the retriever.")
    generated_answer: Optional[str] = Field(
        default=None,
        description="The answer generated based on the retrieval question.")
    progress_bar_id: Optional[str] = Field(
        default=None,
        description="A specific internal identifier of the SQA system to manage the progress bar.")
    question_id: Optional[str] = Field(
        default=None,
        description="An identifier for the question being processed.")
