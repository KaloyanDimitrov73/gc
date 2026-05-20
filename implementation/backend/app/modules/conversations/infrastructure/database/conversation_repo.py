"""
Repository for all conversation and message database operations.
"""
import uuid
import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import ConversationModel, MessageModel

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Async SQLAlchemy repository for conversations and their messages."""

    def __init__(self, db: AsyncSession) -> None:
        """
        Args:
            db: The async SQLAlchemy session used for all database operations.
        """
        self._db = db

    async def list_conversations(self, user_id: str) -> List[ConversationModel]:
        """Return all conversations for *user_id* ordered by most-recently updated first."""
        result = await self._db.execute(
            select(ConversationModel)
            .where(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.updated_at.desc())
        )
        return list(result.scalars().all())

    async def create_conversation(self, user_id: str, title: str = "New Chat") -> ConversationModel:
        """Create and persist a new conversation.

        Args:
            user_id: The owner of the conversation.
            title: Display title for the conversation. Defaults to ``"New Chat"``.

        Returns:
            The freshly committed ``ConversationModel`` instance.
        """
        conversation = ConversationModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._db.add(conversation)
        await self._db.commit()
        await self._db.refresh(conversation)
        return conversation

    async def get_conversation_with_messages(self, conversation_id: str, user_id: str) -> Optional[ConversationModel]:
        """Fetch a conversation and eagerly load its messages in a single query.

        Args:
            conversation_id: UUID string of the target conversation.
            user_id: The requesting user; used to prevent cross-user access.

        Returns:
            The ``ConversationModel`` with ``messages`` populated, or ``None`` if not found.
        """
        result = await self._db.execute(
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def update_title(self, conversation_id: str, user_id: str, title: str) -> bool:
        """Update the title of a conversation owned by *user_id*.

        Returns:
            ``True`` if the row was updated, ``False`` if not found or not owned.
        """
        result = await self._db.execute(
            update(ConversationModel)
            .where(
                ConversationModel.id == conversation_id,
                ConversationModel.user_id == user_id,
            )
            .values(title=title, updated_at=datetime.utcnow())
        )
        await self._db.commit()
        return result.rowcount > 0

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a single conversation (and its cascaded messages) by id.

        Args:
            conversation_id: UUID string of the conversation to delete.
            user_id: The requesting user; only deletes if they own the conversation.

        Returns:
            ``True`` if a row was deleted, ``False`` if the id was not found or not owned.
        """
        result = await self._db.execute(
            delete(ConversationModel).where(
                ConversationModel.id == conversation_id,
                ConversationModel.user_id == user_id,
            )
        )
        await self._db.commit()
        return result.rowcount > 0

    async def delete_all_conversations(self, user_id: str) -> int:
        """Delete every conversation belonging to *user_id*.

        Returns:
            The number of conversations deleted.
        """
        result = await self._db.execute(
            delete(ConversationModel).where(ConversationModel.user_id == user_id)
        )
        await self._db.commit()
        return result.rowcount

    async def add_messages(
        self,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        nodes: Optional[list],
        sources: Optional[list],
    ) -> None:
        """Append a user/assistant message pair to an existing conversation.

        Both messages share the same timestamp. The conversation's ``updated_at``
        column is explicitly touched because SQLAlchemy's ``onupdate`` trigger only
        fires on direct column writes, not on related-row inserts.

        Args:
            conversation_id: UUID string of the target conversation.
            user_content: Raw text of the user's message.
            assistant_content: Raw text of the assistant's reply.
            nodes: Optional graph nodes attached to the assistant message.
            sources: Optional source references attached to the assistant message.
        """
        now = datetime.utcnow()

        user_msg = MessageModel(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=user_content,
            nodes=None,
            sources=None,
            created_at=now,
        )
        assistant_msg = MessageModel(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_content,
            nodes=nodes,
            sources=sources,
            created_at=now,
        )
        self._db.add(user_msg)
        self._db.add(assistant_msg)

        # Explicitly touch updated_at because onupdate only fires on direct column writes
        await self._db.execute(
            update(ConversationModel)
            .where(ConversationModel.id == conversation_id)
            .values(updated_at=now)
        )

        await self._db.commit()
