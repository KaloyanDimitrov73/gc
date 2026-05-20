"""
Shared fixtures for the API integration tests.

Builds a fully isolated FastAPI test environment for the QA and graph-exploration
endpoints.  No real external services (LLM, graph database, etc.) are needed:
the real service implementations are replaced with in-process stubs before the
routers are imported.

Stubs
-----
FakeRetrievalResult
    Mirrors the shape of a real retrieval result so fixtures can construct
    typed responses without importing the production service layer.
FakeRetrievalService
    In-memory stub for ``RetrievalService``.  Records every call so tests
    can assert on arguments, and can be pre-loaded with a fixed result or
    configured to raise an exception.

Fixtures (module isolation)
---------------------------
qa_router_module
    Patches ``sys.modules`` to inject stub dependency/service modules, then
    reloads the QA router so it binds against the stubs.
graph_router_module
    Same pattern for the graph-exploration router.  Depends on
    ``qa_router_module`` to ensure the shared dependency stubs are already
    in place.

Fixtures (app / HTTP client)
-----------------------------
app
    Assembles a ``FastAPI`` application with both routers mounted under
    ``/api/v1`` and rate-limiting disabled via ``dependency_overrides``.
client
    Wraps ``app`` in a ``TestClient`` (synchronous HTTPX client) ready for
    use in individual test functions.

Fixtures (test data / helpers)
-------------------------------
example_nodes
    A small list of ``GraphNode`` objects covering the common node types
    (resource with a connection, resource without connections, literal).
fake_service_factory
    Factory function that creates ``FakeRetrievalService`` instances.
    Keeps test setup concise: ``fake_service_factory(result=…)``.
retrieval_dependency
    The ``get_retrieval_service`` dependency function from the QA router,
    used as a key in ``app.dependency_overrides``.
graph_explore_dependency
    The ``get_graph_explore_service`` dependency function from the graph
    router, used as a key in ``app.dependency_overrides``.
"""
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.contracts.schemas import GraphNode, NodeConnection


@dataclass
class FakeRetrievalResult:
    """Typed container that mirrors a real retrieval result for use in tests."""

    answer: str
    nodes: list[GraphNode]
    sources: list[str]
    guardrails_warning: Optional[str] = None


class FakeRetrievalService:
    """
    In-memory stub for ``RetrievalService``.

    Configure it with a pre-canned ``result`` or an ``error`` to raise.
    Every call to ``ask`` and ``get_node_neighbors`` is recorded so tests can
    assert on the arguments that were passed in.
    """

    def __init__(
        self,
        *,
        result: Optional[FakeRetrievalResult] = None,
        error: Optional[Exception] = None,
        status: Optional[dict[str, Any]] = None,
    ):
        self._result = result
        self._error = error
        self._status = status or {"available": True, "message": "ok"}
        self.calls: list[dict[str, Any]] = []
        self.neighbor_calls: list[dict[str, Any]] = []

    async def ask(self, **kwargs):
        """Record the call and return the configured result (or raise the configured error)."""
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        if self._result is None:
            raise RuntimeError("FakeRetrievalService has no result configured.")
        return self._result

    def get_status(self):
        """Return the configured status dict (defaults to ``{"available": True, "message": "ok"}``)."""
        return self._status

    def get_node_neighbors(self, **kwargs):
        """Record the call and return the node list from the configured result (or raise the configured error)."""
        self.neighbor_calls.append(kwargs)
        if self._error:
            raise self._error
        if self._result is None:
            raise RuntimeError("FakeRetrievalService has no result configured.")
        return self._result.nodes


@pytest.fixture
def qa_router_module(monkeypatch):
    """
    Reload the QA router against stub modules.

    Injects fake ``dependencies`` and ``RetrievalService`` modules into
    ``sys.modules`` before importing the router, so the router never touches
    real infrastructure.  Tests override ``get_retrieval_service`` via
    ``app.dependency_overrides`` to supply a ``FakeRetrievalService``.
    """
    dependencies_module = types.ModuleType("backend.app.core.dependencies")

    def get_retrieval_service():
        raise RuntimeError("get_retrieval_service must be overridden in tests")

    def get_graph_explore_service():
        raise RuntimeError("get_graph_explore_service must be overridden in tests")

    dependencies_module.get_retrieval_service = get_retrieval_service
    dependencies_module.get_graph_explore_service = get_graph_explore_service
    monkeypatch.setitem(sys.modules, "backend.app.core.dependencies", dependencies_module)

    retrieval_service_module = types.ModuleType(
        "backend.app.modules.qa.application.service"
    )

    class RetrievalService:
        pass

    class RetrievalInputRejectedError(RuntimeError):
        pass

    class RetrievalUnavailableError(RuntimeError):
        pass

    retrieval_service_module.RetrievalService = RetrievalService
    retrieval_service_module.RetrievalInputRejectedError = RetrievalInputRejectedError
    retrieval_service_module.RetrievalUnavailableError = RetrievalUnavailableError
    monkeypatch.setitem(
        sys.modules,
        "backend.app.modules.qa.application.service",
        retrieval_service_module,
    )

    module = importlib.import_module("backend.app.modules.qa.api.router")
    return importlib.reload(module)


@pytest.fixture
def graph_router_module(qa_router_module, monkeypatch):
    """
    Reload the graph-exploration router against a stub service module.

    Depends on ``qa_router_module`` to ensure the shared dependency stubs are
    already present in ``sys.modules``.  Injects a fake
    ``GraphExploreService`` module so the router binds against the stub.
    """
    graph_service_module = types.ModuleType(
        "backend.app.modules.graph_explore.application.graph_explore_service"
    )

    class GraphExploreService:
        pass

    class GraphExploreUnavailableError(RuntimeError):
        pass

    graph_service_module.GraphExploreService = GraphExploreService
    graph_service_module.GraphExploreUnavailableError = GraphExploreUnavailableError
    monkeypatch.setitem(
        sys.modules,
        "backend.app.modules.graph_explore.application.graph_explore_service",
        graph_service_module,
    )

    module = importlib.import_module("backend.app.modules.graph_explore.api.router")
    return importlib.reload(module)


@pytest.fixture
def app(qa_router_module, graph_router_module):
    """
    Assemble a FastAPI app with both routers and rate-limiting disabled.

    Both routers are mounted under ``/api/v1``.  The ``enforce_rate_limit``
    dependency is overridden with a no-op so tests are not throttled.
    """
    fastapi_app = FastAPI()
    fastapi_app.include_router(qa_router_module.router, prefix="/api/v1")
    fastapi_app.include_router(graph_router_module.router, prefix="/api/v1")
    fastapi_app.dependency_overrides[qa_router_module.enforce_rate_limit] = lambda: None
    fastapi_app.dependency_overrides[graph_router_module.enforce_rate_limit] = lambda: None
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    """Synchronous HTTP test client pointed at the assembled app."""
    with TestClient(app, base_url="http://localhost") as test_client:
        yield test_client


@pytest.fixture
def example_nodes():
    """
    Small list of ``GraphNode`` objects covering the common node types.

    Includes a resource node with an outgoing connection, a resource node
    without connections, and a literal node (``depth=None``).
    """
    return [
        GraphNode(
            id="R1",
            label="Paper A",
            type="resource",
            depth=0,
            connections=[NodeConnection(targetId="R2", relation="cites")],
        ),
        GraphNode(id="R2", label="Paper B", type="resource", depth=2, connections=[]),
        GraphNode(id="L1", label="Literal", type="literal", depth=None, connections=[]),
    ]


@pytest.fixture
def fake_service_factory():
    """
    Factory for ``FakeRetrievalService`` instances.

    Returns a callable so individual tests can create stubs inline::

        service = fake_service_factory(result=FakeRetrievalResult(...))
    """
    def factory(*, result=None, error=None, status=None):
        return FakeRetrievalService(result=result, error=error, status=status)

    return factory


@pytest.fixture
def retrieval_dependency(qa_router_module):
    """The ``get_retrieval_service`` function from the QA router, for use as a key in ``app.dependency_overrides``."""
    return qa_router_module.get_retrieval_service


@pytest.fixture
def graph_explore_dependency(graph_router_module):
    """The ``get_graph_explore_service`` function from the graph router, for use as a key in ``app.dependency_overrides``."""
    return graph_router_module.get_graph_explore_service
