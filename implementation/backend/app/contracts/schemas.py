"""
Pydantic models for API request/response validation.
These schemas define the contract between the frontend and backend.
"""
from datetime import datetime
from typing import Any, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field


class NodeConnection(BaseModel):
    """Represents a connection between two nodes in the knowledge graph."""
    target_id: str = Field(..., alias="targetId", description="ID of the target node")
    relation: str = Field(..., description="Type of relation between nodes")

    class Config:
        populate_by_name = True


class GraphNode(BaseModel):
    """Represents a node in the knowledge graph visualization."""
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Display label for the node")
    type: str = Field(..., description="Type of node ('resource' or 'literal')")
    connections: List[NodeConnection] = Field(default_factory=list, description="Connections to other nodes")
    x: Optional[float] = Field(None, description="X coordinate for positioning")
    y: Optional[float] = Field(None, description="Y coordinate for positioning")
    score: Optional[float] = Field(None, description="Relevance score")

    class Config:
        populate_by_name = True


class QuestionRequest(BaseModel):
    """Request payload for asking a question."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question"
    )
    retrieval_mode: Literal['direct', 'graph'] = Field(
        'direct',
        alias="retrievalMode",
        description="Retrieval strategy: 'direct' for vector-based, 'graph' for traversal-based"
    )
    llm_model: str = Field(
        'gpt-4o-mini',
        alias="llmModel",
        max_length=128,
        description="LLM model to use for answer generation"
    )
    number_of_hubs: int = Field(
        10,
        alias="numberOfHubs",
        ge=1,
        le=50,
        description="Number of hub entities to retrieve and analyze"
    )
    topic_entity_id: Optional[str] = Field(
        None,
        alias="topicEntityId",
        max_length=128,
        description="Optional topic entity ID for graph traversal mode"
    )
    use_direct_final_answer: bool = Field(
        False,
        alias="useDirectFinalAnswer",
        description="When True, hub paths are fed directly into the final answer prompt, "
                    "skipping per-hub partial answer generation."
    )
    conversation_id: Optional[str] = Field(
        None,
        alias="conversationId",
        max_length=36,
        description="Conversation ID to persist this exchange under"
    )

    class Config:
        populate_by_name = True


class AnswerResponse(BaseModel):
    """Response containing the generated answer and knowledge graph."""
    answer: str = Field(..., description="Generated answer to the question")
    nodes: List[GraphNode] = Field(default_factory=list, description="Knowledge graph nodes")
    retrieval_mode: str = Field(..., alias="retrievalMode", description="Retrieval mode used")
    sources: List[str] = Field(default_factory=list, description="Source identifiers (DOIs, URLs)")
    message_id: str = Field(..., alias="messageId", description="Unique message identifier")
    guardrails_warning: Optional[str] = Field(
        None,
        alias="guardrailsWarning",
        description="Warning message if output validation flagged issues"
    )

    class Config:
        populate_by_name = True


class NodeNeighborsResponse(BaseModel):
    """Graph fragment returned when expanding a node in the UI."""
    node_id: str = Field(..., alias="nodeId", description="Expanded node identifier")
    nodes: List[GraphNode] = Field(default_factory=list, description="Graph fragment centered on the node")

    class Config:
        populate_by_name = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    hublink_available: bool = Field(..., alias="hublinkAvailable")
    message: str

    class Config:
        populate_by_name = True


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    fallback_used: bool = Field(False, alias="fallbackUsed", description="Whether fallback was used")

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# Conversation management schemas
# ---------------------------------------------------------------------------

class CreateConversationRequest(BaseModel):
    """Request payload to create a new chat conversation."""
    title: Optional[str] = Field(None, max_length=255, description="Conversation title; defaults to 'New Chat'")


class UpdateConversationRequest(BaseModel):
    """Request payload to update an existing conversation."""
    title: str = Field(..., min_length=1, max_length=255, description="New conversation title")


class ConversationSchema(BaseModel):
    """Chat conversation metadata."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: str
    title: str
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")


class MessageSchema(BaseModel):
    """A single persisted chat message."""
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: str
    conversation_id: str = Field(..., alias="conversationId")
    role: str
    content: str
    nodes: Optional[Any] = None
    sources: Optional[List[str]] = None
    created_at: datetime = Field(..., alias="createdAt")


class ConversationDetailSchema(ConversationSchema):
    """Chat conversation metadata together with all its messages."""
    messages: List[MessageSchema] = Field(default_factory=list)
