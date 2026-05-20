'''All Depends() factories'''

from typing import Optional
import threading

from backend.app.modules.qa.infrastructure.hublink.hublink_service import HubLinkService
from backend.app.modules.qa.infrastructure.hublink.config_loader import HublinkConfigLoader
from backend.app.modules.qa.infrastructure.hublink.llm_config_registry import LLMConfigRegistry
from backend.app.modules.guardrails.application.service import GuardrailsService
from backend.app.modules.qa.application.service import RetrievalService
from backend.app.modules.graph_load.application.graph_load_service import GraphLoadService
from backend.app.modules.graph_explore.application.graph_explore_service import GraphExploreService

_graph_load_lock = threading.Lock()
_graph_explore_lock = threading.Lock()
_hublink_lock = threading.Lock()
_guardrails_lock = threading.Lock()
_retrieval_lock = threading.Lock()

_graph_load_service: Optional[GraphLoadService] = None
_graph_explore_service: Optional[GraphExploreService] = None
_hublink_service: Optional[HubLinkService] = None
_guardrails_service: Optional[GuardrailsService] = None
_retrieval_service: Optional[RetrievalService] = None
_llm_config_registry: Optional[LLMConfigRegistry] = None


def get_graph_load_service() -> GraphLoadService:
    """Return the singleton ``GraphLoadService``, creating it on first call.

    Loads the KG retrieval config via ``HublinkConfigLoader`` and initialises
    the service. Thread-safe via a double-checked locking pattern.
    """
    global _graph_load_service
    if _graph_load_service is None:
        with _graph_load_lock:
            if _graph_load_service is None:
                config = HublinkConfigLoader().load_kg_retrieval_config()
                _graph_load_service = GraphLoadService(config=config)
    return _graph_load_service


def get_graph_explore_service() -> GraphExploreService:
    """Return the singleton ``GraphExploreService``, creating it on first call.

    Depends on ``get_graph_load_service`` to share the already-loaded graph.
    Thread-safe via a double-checked locking pattern.
    """
    global _graph_explore_service
    if _graph_explore_service is None:
        with _graph_explore_lock:
            if _graph_explore_service is None:
                graph_load = get_graph_load_service()
                _graph_explore_service = GraphExploreService(graph=graph_load.graph)
    return _graph_explore_service


def get_hublink_service() -> HubLinkService:
    """Return the singleton ``HubLinkService``, creating it on first call.

    Shares the graph loaded by ``get_graph_load_service``.
    Thread-safe via a double-checked locking pattern.
    """
    global _hublink_service
    if _hublink_service is None:
        with _hublink_lock:
            if _hublink_service is None:
                graph_load = get_graph_load_service()
                _hublink_service = HubLinkService(graph=graph_load.graph)
    return _hublink_service


def get_guardrails_service() -> GuardrailsService:
    """Return the singleton ``GuardrailsService``, creating it on first call.

    Registers ``ToxicLanguageValidator`` and ``JailbreakValidator`` for both
    input and output validation. ``GibberishValidator`` is intentionally
    excluded due to too many false negatives on domain queries.
    Thread-safe via a double-checked locking pattern.
    """
    global _guardrails_service
    if _guardrails_service is None:
        with _guardrails_lock:
            if _guardrails_service is None:
                from backend.app.modules.guardrails.application.validators import (
                    ToxicLanguageValidator,
                    JailbreakValidator,
                    # GibberishValidator excluded: too many false negatives on domain queries
                )
                validators = [ToxicLanguageValidator(), JailbreakValidator()]
                _guardrails_service = GuardrailsService(
                    input_validators=validators,
                    output_validators=validators,
                )
    return _guardrails_service


def get_retrieval_service() -> RetrievalService:
    """Return the singleton ``RetrievalService``, creating it on first call.

    Composes ``HubLinkService`` and ``GuardrailsService``.
    Thread-safe via a double-checked locking pattern.
    """
    global _retrieval_service
    if _retrieval_service is None:
        hublink = get_hublink_service()
        guardrails = get_guardrails_service()
        with _retrieval_lock:
            if _retrieval_service is None:
                _retrieval_service = RetrievalService(
                    hublink_service=hublink,
                    guardrails_service=guardrails,
                )
    return _retrieval_service


def get_retrieval_service_if_ready() -> "RetrievalService | None":
    """Return the singleton ``RetrievalService`` only if already initialized.

    Does NOT trigger initialization. Returns ``None`` when services are still loading.
    """
    return _retrieval_service


# ConversationService is request-scoped (depends on a per-request AsyncSession) —
# no singleton pattern needed.
from fastapi import Depends  # noqa: E402 — placed here to avoid circular imports
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.modules.conversations.infrastructure.database.db import get_db
from backend.app.modules.conversations.infrastructure.database.conversation_repo import ConversationRepository
from backend.app.modules.conversations.application.service import ConversationService


def get_conversation_service(db: AsyncSession = Depends(get_db)) -> ConversationService:
    """FastAPI dependency that creates a request-scoped ``ConversationService``.

    A new instance is created per request because it wraps a per-request
    ``AsyncSession`` — no singleton pattern is used here.
    """
    return ConversationService(ConversationRepository(db))


def get_llm_config_registry() -> LLMConfigRegistry:
    """Return the singleton ``LLMConfigRegistry``, creating it on first call."""
    global _llm_config_registry
    if _llm_config_registry is None:
        _llm_config_registry = LLMConfigRegistry()
    return _llm_config_registry
