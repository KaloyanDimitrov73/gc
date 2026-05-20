"""
Unit tests for ConversationService.

Verifies schema mapping, business logic, and the persist_qa_exchange contract.
All tests run against an isolated in-memory SQLite database (see conftest.py).
"""
import pytest

from backend.app.contracts.schemas import ConversationDetailSchema, ConversationSchema
from backend.app.modules.conversations.application.service import ConversationService
from backend.app.modules.conversations.infrastructure.database.conversation_repo import ConversationRepository

pytestmark = pytest.mark.unit


def _make_svc(db_session) -> ConversationService:
    return ConversationService(ConversationRepository(db_session))


@pytest.mark.asyncio
async def test_create_conversation_returns_conversation_schema(db_session):
    result = await _make_svc(db_session).create_conversation()
    assert isinstance(result, ConversationSchema)
    assert result.title == "New Chat"
    assert result.id is not None


@pytest.mark.asyncio
async def test_create_conversation_with_title(db_session):
    result = await _make_svc(db_session).create_conversation(title="Custom Title")
    assert result.title == "Custom Title"


@pytest.mark.asyncio
async def test_list_conversations_returns_list_of_schemas(db_session):
    svc = _make_svc(db_session)
    await svc.create_conversation("A")
    await svc.create_conversation("B")
    conversations = await svc.list_conversations()
    assert len(conversations) == 2
    assert all(isinstance(c, ConversationSchema) for c in conversations)


@pytest.mark.asyncio
async def test_list_conversations_empty(db_session):
    conversations = await _make_svc(db_session).list_conversations()
    assert conversations == []


@pytest.mark.asyncio
async def test_get_conversation_detail_returns_detail_schema(db_session):
    svc = _make_svc(db_session)
    conversation = await svc.create_conversation("detail test")
    result = await svc.get_conversation_detail(conversation.id)
    assert isinstance(result, ConversationDetailSchema)
    assert result.id == conversation.id
    assert result.messages == []


@pytest.mark.asyncio
async def test_get_conversation_detail_not_found_returns_none(db_session):
    result = await _make_svc(db_session).get_conversation_detail("missing")
    assert result is None


@pytest.mark.asyncio
async def test_persist_qa_exchange_creates_two_messages(db_session):
    svc = _make_svc(db_session)
    conversation = await svc.create_conversation("qa test")
    nodes = [{"id": "n1", "label": "Paper", "type": "resource", "connections": []}]

    await svc.persist_qa_exchange(
        conversation_id=conversation.id,
        question="What is X?",
        answer="X is Y.",
        nodes=nodes,
        sources=["doi:10.1000/xyz"],
    )

    detail = await svc.get_conversation_detail(conversation.id)
    assert len(detail.messages) == 2

    user_msg = detail.messages[0]
    assert user_msg.role == "user"
    assert user_msg.content == "What is X?"

    assistant_msg = detail.messages[1]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "X is Y."
    assert assistant_msg.nodes == nodes
    assert assistant_msg.sources == ["doi:10.1000/xyz"]


@pytest.mark.asyncio
async def test_persist_qa_exchange_without_nodes_and_sources(db_session):
    svc = _make_svc(db_session)
    conversation = await svc.create_conversation()
    await svc.persist_qa_exchange(conversation.id, "q", "a", None, None)
    detail = await svc.get_conversation_detail(conversation.id)
    assistant_msg = detail.messages[1]
    assert assistant_msg.nodes is None
    assert assistant_msg.sources is None


@pytest.mark.asyncio
async def test_delete_conversation_returns_true(db_session):
    svc = _make_svc(db_session)
    conversation = await svc.create_conversation()
    assert await svc.delete_conversation(conversation.id) is True


@pytest.mark.asyncio
async def test_delete_conversation_removes_from_list(db_session):
    svc = _make_svc(db_session)
    conversation = await svc.create_conversation()
    await svc.delete_conversation(conversation.id)
    assert await svc.get_conversation_detail(conversation.id) is None
    assert await svc.list_conversations() == []


@pytest.mark.asyncio
async def test_delete_nonexistent_conversation_returns_false(db_session):
    assert await _make_svc(db_session).delete_conversation("ghost") is False
