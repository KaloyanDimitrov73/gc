"""
Conversation management API endpoints.
Provides CRUD operations for chat conversations and their persisted messages.
"""
import logging
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from backend.app.contracts.schemas import (
    CreateConversationRequest,
    UpdateConversationRequest,
    ConversationSchema,
    ConversationDetailSchema,
)
from backend.app.core.dependencies import get_conversation_service, get_or_create_user_id
from backend.app.modules.conversations.application.service import ConversationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])

@router.get(
    "/",
    response_model=List[ConversationSchema],
    summary="List all chat conversations",
    description="Returns all conversations for the current user ordered by most recently updated first.",
)
async def list_conversations(
    user_id: str = Depends(get_or_create_user_id),
    conversation_svc: ConversationService = Depends(get_conversation_service),
) -> List[ConversationSchema]:
    return await conversation_svc.list_conversations(user_id=user_id)


@router.post(
    "/",
    response_model=ConversationSchema,
    status_code=201,
    summary="Create a new chat conversation",
    description="Creates a new conversation for the current user. Title defaults to 'New Chat' if not provided.",
)
async def create_conversation(
    body: CreateConversationRequest,
    user_id: str = Depends(get_or_create_user_id),
    conversation_svc: ConversationService = Depends(get_conversation_service),
) -> ConversationSchema:
    return await conversation_svc.create_conversation(user_id=user_id, title=body.title)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationSchema,
    summary="Update a conversation",
    description="Updates the title of the specified conversation.",
)
async def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    user_id: str = Depends(get_or_create_user_id),
    conversation_svc: ConversationService = Depends(get_conversation_service),
) -> ConversationSchema:
    updated = await conversation_svc.update_title(conversation_id, user_id=user_id, title=body.title)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    detail = await conversation_svc.get_conversation_detail(conversation_id, user_id=user_id)
    return detail


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailSchema,
    summary="Get a conversation with its messages",
    description="Returns the conversation metadata and all persisted messages in chronological order.",
)
async def get_conversation(
    conversation_id: str,
    user_id: str = Depends(get_or_create_user_id),
    conversation_svc: ConversationService = Depends(get_conversation_service),
) -> ConversationDetailSchema:
    detail = await conversation_svc.get_conversation_detail(conversation_id, user_id=user_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    return detail


@router.delete(
    "/",
    status_code=204,
    summary="Delete all conversations",
    description="Deletes every conversation belonging to the current user.",
)
async def delete_all_conversations(
    user_id: str = Depends(get_or_create_user_id),
    conversation_svc: ConversationService = Depends(get_conversation_service),
) -> None:
    await conversation_svc.delete_all_conversations(user_id=user_id)


@router.delete(
    "/{conversation_id}",
    status_code=204,
    summary="Delete a conversation",
    description="Deletes the conversation and all its messages (cascade).",
)
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_or_create_user_id),
    conversation_svc: ConversationService = Depends(get_conversation_service),
) -> None:
    deleted = await conversation_svc.delete_conversation(conversation_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
