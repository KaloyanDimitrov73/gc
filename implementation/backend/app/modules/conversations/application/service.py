"""
Conversation application service.
Maps between ORM models and Pydantic schemas; orchestrates repository calls.
"""
import logging
from typing import List, Optional

from backend.app.contracts.schemas import (
    ConversationSchema,
    ConversationDetailSchema,
    MessageSchema,
)
from backend.app.modules.conversations.infrastructure.database.models import (
    ConversationModel,
    MessageModel,
)
from backend.app.modules.conversations.infrastructure.database.conversation_repo import ConversationRepository

logger = logging.getLogger(__name__)


def _map_message(msg: MessageModel) -> MessageSchema:
    """Map a ``MessageModel`` ORM row to a ``MessageSchema`` Pydantic model."""
    return MessageSchema(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        nodes=msg.nodes,
        sources=msg.sources,
        created_at=msg.created_at,
    )


def _map_conversation(conversation: ConversationModel) -> ConversationSchema:
    """Map a ``ConversationModel`` ORM row to a ``ConversationSchema`` Pydantic model."""
    return ConversationSchema(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _map_conversation_detail(conversation: ConversationModel) -> ConversationDetailSchema:
    """Map a ``ConversationModel`` (with eagerly loaded messages) to a ``ConversationDetailSchema``."""
    return ConversationDetailSchema(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[_map_message(m) for m in conversation.messages],
    )


class ConversationService:
    """Application service for conversation CRUD and message persistence."""

    def __init__(self, repo: ConversationRepository) -> None:
        """
        Args:
            repo: The repository used for all database operations.
        """
        self._repo = repo

    async def list_conversations(self, user_id: str) -> List[ConversationSchema]:
        """Return all conversations for *user_id* ordered by most-recently updated first."""
        conversations = await self._repo.list_conversations(user_id=user_id)
        return [_map_conversation(c) for c in conversations]

    async def create_conversation(self, user_id: str, title: Optional[str] = None) -> ConversationSchema:
        """Create and return a new conversation.

        Args:
            user_id: The owner of the new conversation.
            title: Optional display title; defaults to ``"New Chat"``.
        """
        conversation = await self._repo.create_conversation(user_id=user_id, title=title or "New Chat")
        return _map_conversation(conversation)

    async def get_conversation_detail(self, conversation_id: str, user_id: str) -> Optional[ConversationDetailSchema]:
        """Return a conversation with its full message history, or ``None`` if not found.

        Args:
            conversation_id: UUID string of the target conversation.
            user_id: The requesting user; access is denied if they do not own the conversation.
        """
        conversation = await self._repo.get_conversation_with_messages(conversation_id, user_id=user_id)
        if conversation is None:
            return None
        return _map_conversation_detail(conversation)

    async def update_title(self, conversation_id: str, user_id: str, title: str) -> bool:
        """Update the display title of a conversation.

        Returns:
            ``True`` if updated, ``False`` if not found or not owned.
        """
        return await self._repo.update_title(conversation_id, user_id=user_id, title=title)

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete the conversation identified by *conversation_id* if owned by *user_id*.

        Returns:
            ``True`` if a row was deleted, ``False`` if the id was not found or not owned.
        """
        return await self._repo.delete_conversation(conversation_id, user_id=user_id)

    async def delete_all_conversations(self, user_id: str) -> int:
        """Delete every conversation owned by *user_id*.

        Returns:
            The number of conversations deleted.
        """
        return await self._repo.delete_all_conversations(user_id=user_id)

    async def persist_qa_exchange(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        nodes: Optional[list],
        sources: Optional[list],
    ) -> None:
        """Persist one user message and one assistant message for a completed Q&A exchange."""
        await self._repo.add_messages(
            conversation_id=conversation_id,
            user_content=question,
            assistant_content=answer,
            nodes=nodes,
            sources=sources,
        )
        logger.debug("Persisted QA exchange for conversation %s", conversation_id)
