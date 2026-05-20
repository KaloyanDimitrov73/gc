"""
Integration tests for the conversation management HTTP endpoints.

Uses a minimal FastAPI app with the conversations router and an in-memory SQLite
database injected via dependency_overrides — no external services needed.

Run together with the other integration API tests:
    pytest tests/integration/api/ -v
"""
import asyncio

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.contracts.schemas import CreateConversationRequest
from backend.app.core.dependencies import get_conversation_service
from backend.app.modules.conversations.api.router import router as conversations_router
from backend.app.modules.conversations.application.service import ConversationService
from backend.app.modules.conversations.infrastructure.database.models import Base
from backend.app.modules.conversations.infrastructure.database.conversation_repo import ConversationRepository

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


def _build_db():
    """Create an in-memory engine and run table creation synchronously."""
    engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _init():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.get_event_loop().run_until_complete(_init())
    return engine, factory


@pytest.fixture
def client():
    engine, factory = _build_db()

    def override_conversation_service():
        # Each request gets a fresh session from the in-memory DB
        session = factory()

        async def _close():
            await session.close()

        # Return a service backed by the shared in-memory engine
        return ConversationService(ConversationRepository(session))

    app = FastAPI()
    app.include_router(conversations_router, prefix="/api/v1")
    app.dependency_overrides[get_conversation_service] = override_conversation_service

    with TestClient(app, base_url="http://localhost") as c:
        yield c

    app.dependency_overrides.clear()
    asyncio.get_event_loop().run_until_complete(engine.dispose())


# ---------------------------------------------------------------------------
# GET /api/v1/conversations/
# ---------------------------------------------------------------------------

def test_list_conversations_empty(client):
    r = client.get("/api/v1/conversations/")
    assert r.status_code == 200
    assert r.json() == []


def test_list_conversations_returns_created_conversations(client):
    client.post("/api/v1/conversations/", json={"title": "Alpha"})
    client.post("/api/v1/conversations/", json={"title": "Beta"})
    r = client.get("/api/v1/conversations/")
    assert r.status_code == 200
    titles = {c["title"] for c in r.json()}
    assert titles == {"Alpha", "Beta"}


# ---------------------------------------------------------------------------
# POST /api/v1/conversations/
# ---------------------------------------------------------------------------

def test_create_conversation_default_title(client):
    r = client.post("/api/v1/conversations/", json={})
    assert r.status_code == 201
    data = r.json()
    assert data["title"] == "New Chat"
    assert "id" in data
    assert "createdAt" in data
    assert "updatedAt" in data


def test_create_conversation_with_title(client):
    r = client.post("/api/v1/conversations/", json={"title": "My research chat"})
    assert r.status_code == 201
    assert r.json()["title"] == "My research chat"


def test_create_conversation_title_max_length_enforced(client):
    long_title = "x" * 256  # max is 255
    r = client.post("/api/v1/conversations/", json={"title": long_title})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/conversations/{conversation_id}
# ---------------------------------------------------------------------------

def test_get_conversation_returns_empty_messages(client):
    conversation_id = client.post("/api/v1/conversations/", json={}).json()["id"]
    r = client.get(f"/api/v1/conversations/{conversation_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == conversation_id
    assert data["messages"] == []


def test_get_conversation_not_found(client):
    r = client.get("/api/v1/conversations/nonexistent-id")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# DELETE /api/v1/conversations/{conversation_id}
# ---------------------------------------------------------------------------

def test_delete_conversation_returns_204(client):
    conversation_id = client.post("/api/v1/conversations/", json={}).json()["id"]
    r = client.delete(f"/api/v1/conversations/{conversation_id}")
    assert r.status_code == 204


def test_delete_conversation_makes_it_unfindable(client):
    conversation_id = client.post("/api/v1/conversations/", json={}).json()["id"]
    client.delete(f"/api/v1/conversations/{conversation_id}")
    assert client.get(f"/api/v1/conversations/{conversation_id}").status_code == 404


def test_delete_nonexistent_conversation_returns_404(client):
    r = client.delete("/api/v1/conversations/ghost-id")
    assert r.status_code == 404


def test_delete_removes_conversation_from_list(client):
    conversation_id = client.post("/api/v1/conversations/", json={"title": "to delete"}).json()["id"]
    client.post("/api/v1/conversations/", json={"title": "keep"})
    client.delete(f"/api/v1/conversations/{conversation_id}")
    remaining = client.get("/api/v1/conversations/").json()
    assert len(remaining) == 1
    assert remaining[0]["title"] == "keep"
