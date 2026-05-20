"""
Unit tests for ConversationRepository.

All tests run against an isolated in-memory SQLite database (see conftest.py).
No external services, credentials, or network access required.
"""
import pytest

from backend.app.modules.conversations.infrastructure.database.conversation_repo import ConversationRepository

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_list_conversations_initially_empty(db_session):
    repo = ConversationRepository(db_session)
    assert await repo.list_conversations() == []


@pytest.mark.asyncio
async def test_create_conversation_default_title(db_session):
    repo = ConversationRepository(db_session)
    conversation = await repo.create_conversation()
    assert conversation.title == "New Chat"
    assert conversation.id is not None
    assert conversation.created_at is not None
    assert conversation.updated_at is not None


@pytest.mark.asyncio
async def test_create_conversation_custom_title(db_session):
    repo = ConversationRepository(db_session)
    conversation = await repo.create_conversation(title="My Research Question")
    assert conversation.title == "My Research Question"


@pytest.mark.asyncio
async def test_list_conversations_returns_all_created(db_session):
    repo = ConversationRepository(db_session)
    await repo.create_conversation(title="Alpha")
    await repo.create_conversation(title="Beta")
    conversations = await repo.list_conversations()
    assert len(conversations) == 2
    titles = {c.title for c in conversations}
    assert titles == {"Alpha", "Beta"}


@pytest.mark.asyncio
async def test_list_conversations_ordered_by_updated_at_descending(db_session):
    repo = ConversationRepository(db_session)
    c1 = await repo.create_conversation(title="Older")
    await repo.create_conversation(title="Newer")
    # Persisting messages bumps c1.updated_at, making it the newest
    await repo.add_messages(c1.id, "q", "a", None, None, None)
    conversations = await repo.list_conversations()
    assert conversations[0].id == c1.id


@pytest.mark.asyncio
async def test_get_conversation_with_messages_empty(db_session):
    repo = ConversationRepository(db_session)
    conversation = await repo.create_conversation()
    result = await repo.get_conversation_with_messages(conversation.id)
    assert result is not None
    assert result.id == conversation.id
    assert result.messages == []


@pytest.mark.asyncio
async def test_get_conversation_with_messages_persists_both_roles(db_session):
    repo = ConversationRepository(db_session)
    conversation = await repo.create_conversation()
    nodes = [{"id": "n1", "label": "Node", "type": "resource", "connections": []}]
    await repo.add_messages(conversation.id, "What is X?", "X is Y.", nodes, ["doi:1"])

    result = await repo.get_conversation_with_messages(conversation.id)
    assert len(result.messages) == 2
    user_msg, assistant_msg = result.messages[0], result.messages[1]

    assert user_msg.role == "user"
    assert user_msg.content == "What is X?"
    assert user_msg.nodes is None

    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "X is Y."
    assert assistant_msg.nodes == nodes
    assert assistant_msg.sources == ["doi:1"]


@pytest.mark.asyncio
async def test_get_conversation_not_found_returns_none(db_session):
    repo = ConversationRepository(db_session)
    result = await repo.get_conversation_with_messages("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_delete_conversation_returns_true_and_removes_record(db_session):
    repo = ConversationRepository(db_session)
    conversation = await repo.create_conversation()
    deleted = await repo.delete_conversation(conversation.id)
    assert deleted is True
    assert await repo.get_conversation_with_messages(conversation.id) is None


@pytest.mark.asyncio
async def test_delete_conversation_cascades_to_messages(db_session):
    repo = ConversationRepository(db_session)
    conversation = await repo.create_conversation()
    await repo.add_messages(conversation.id, "q", "a", None, None)
    # Verify messages exist
    result = await repo.get_conversation_with_messages(conversation.id)
    assert len(result.messages) == 2
    # Delete conversation — messages must be gone too (cascade)
    await repo.delete_conversation(conversation.id)
    assert await repo.get_conversation_with_messages(conversation.id) is None


@pytest.mark.asyncio
async def test_delete_nonexistent_conversation_returns_false(db_session):
    repo = ConversationRepository(db_session)
    assert await repo.delete_conversation("ghost-id") is False


@pytest.mark.asyncio
async def test_add_messages_without_nodes_or_sources(db_session):
    repo = ConversationRepository(db_session)
    conversation = await repo.create_conversation()
    await repo.add_messages(conversation.id, "hello", "world", None, None, None)
    result = await repo.get_conversation_with_messages(conversation.id)
    assistant = result.messages[1]
    assert assistant.nodes is None
    assert assistant.sources is None
