# Conversations Module

Handles all conversation and message persistence for the chat interface.

---

## Design

Each conversation is an independent unit of state. The database holds two linked tables: `conversations` (one row per chat) and `messages` (one row per turn, linked to its conversation by `conversation_id`). Deleting a conversation cascade-deletes all its messages.

### How a conversation is created

A conversation can be created in two ways:

1. **Explicitly** — the user clicks "New Chat". The frontend calls `POST /api/v1/conversations/`, which inserts a row with a UUID and the title `"New Chat"`.
2. **Implicitly** — the user sends a first message without an active conversation. The `qa` module creates one on the fly (using the first 50 characters of the question as the title) before processing the question.

### How messages are persisted

After HubLink returns an answer, the `qa` module calls `ConversationService.persist_qa_exchange()`. This writes the user turn and the assistant turn as a pair in a single transaction, and updates `conversations.updated_at` so the sidebar stays sorted by most-recent activity.

The assistant message stores two additional fields alongside the text:
- `nodes` — the knowledge-graph nodes HubLink used (JSON)
- `sources` — source identifiers

### How the frontend uses this

The frontend maintains a per-conversation state bucket in memory (`Record<conversationId, { messages, isLoading, selectedMessageId }>`). When the user switches to a conversation whose messages are not yet cached locally, the frontend fetches `GET /api/v1/conversations/{id}` to load the full message history. Subsequent navigation is served from the local cache; only a new Q&A exchange triggers another write.

This separation means a slow in-flight request always updates its own conversation's bucket — navigating away during a request does not cancel it or corrupt another conversation's state.

---

## Limitations and known trade-offs

- **No user ownership.** Every conversation is global — there is no `user_id` column. All conversations are visible and modifiable by any client.
- **`create_all()` cannot evolve the schema.** If a column is renamed or added in the ORM model, the existing database file is unaffected. The only current workaround is to delete `backend/data/hublink.db` and let `init_db()` recreate it on the next startup.
- **SQLite is single-writer.** Concurrent writes are serialised. This is acceptable for a single-user development setup but will become a bottleneck under multi-user load.

---

## Possible suggestions before adding user management: migrate to PostgreSQL + Alembic

The current storage choices (SQLite, `create_all()`) are deliberate shortcuts for a prototype. They must be replaced before real users are added:

1. **Switch to PostgreSQL** — eliminates the single-writer limit and supports connection pooling, row-level locks, and concurrent access.

2. **Introduce Alembic** — replaces the `create_all()` shortcut with versioned, reversible migrations. Schema changes become `ALTER TABLE` statements instead of drop-and-recreate. Set up with `alembic init alembic` and point `env.py` at the async engine.

3. **Add a nullable `user_id` column to `conversations`** — links each conversation to its owner. All queries should be scoped to the authenticated user's ID.

4. **Use JWT for authentication** — issue a signed token at login and verify it per request. No server-side session table is needed.

Suggested migration path: `SQLite → PostgreSQL`, `create_all() → Alembic`, `no auth → JWT`.
