"""
Integration tests for the QA router's happy-path behaviour.

Verifies that POST /api/v1/qa/ask serialises the service result correctly,
that GET /api/v1/qa/health and POST /api/v1/qa/init reflect service status,
and that GET /api/v1/graph/node-neighbors returns the expected graph fragment.

All real service implementations are replaced via app.dependency_overrides —
no LLM or graph-database connections are made.
"""
from types import SimpleNamespace


def test_ask_returns_answer_payload(
    app, client, example_nodes, fake_service_factory, retrieval_dependency
):
    """Full round-trip: verifies the response shape, field values, and that the service received the exact call arguments."""
    fake_service = fake_service_factory(
        result=SimpleNamespace(
            answer="Robustness is investigated by Paper A.",
            nodes=example_nodes,
            sources=["10.1000/example-doi"],
            guardrails_warning="Potentially uncertain statement.",
        )
    )
    app.dependency_overrides[retrieval_dependency] = lambda: fake_service

    response = client.post(
        "/api/v1/qa/ask",
        json={
            "question": "Which paper investigates robustness?",
            "retrievalMode": "graph",
            "llmModel": "qwen3:14b",
            "numberOfHubs": 12,
            "topicEntityId": "R123",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["answer"] == "Robustness is investigated by Paper A."
    assert payload["retrievalMode"] == "graph"
    assert payload["guardrailsWarning"] == "Potentially uncertain statement."
    assert payload["messageId"].startswith("msg-")
    assert len(payload["nodes"]) == 3
    assert payload["nodes"][0]["connections"][0]["targetId"] == "R2"
    assert payload["sources"] == ["10.1000/example-doi"]

    assert fake_service.calls == [
        {
            "question": "Which paper investigates robustness?",
            "retrieval_mode": "graph",
            "llm_model": "qwen3:14b",
            "number_of_hubs": 12,
            "topic_entity_id": "R123",
        }
    ]


def test_ask_uses_defaults_when_optional_fields_missing(
    app, client, example_nodes, fake_service_factory, retrieval_dependency
):
    """Omitting optional fields (retrievalMode, llmModel, etc.) should apply documented defaults."""
    fake_service = fake_service_factory(
        result=SimpleNamespace(
            answer="Default path result.",
            nodes=example_nodes,
            sources=[],
            guardrails_warning=None,
        )
    )
    app.dependency_overrides[retrieval_dependency] = lambda: fake_service

    response = client.post("/api/v1/qa/ask", json={"question": "Simple question"})

    assert response.status_code == 200
    assert response.json()["retrievalMode"] == "direct"

    assert fake_service.calls == [
        {
            "question": "Simple question",
            "retrieval_mode": "direct",
            "llm_model": "gpt-4o-mini",
            "number_of_hubs": 10,
            "topic_entity_id": None,
        }
    ]


def test_health_and_init_return_service_status(
    app, client, fake_service_factory, retrieval_dependency
):
    """Both the health-check and init endpoints should surface the service's status dict."""
    fake_service = fake_service_factory(status={"available": True, "message": "HubLink ready"})
    app.dependency_overrides[retrieval_dependency] = lambda: fake_service

    health_response = client.get("/api/v1/qa/health")
    init_response = client.post("/api/v1/qa/init")

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"
    assert health_response.json()["hublinkAvailable"] is True
    assert health_response.json()["hublink_available"] is True
    assert health_response.json()["message"] == "HubLink ready"

    assert init_response.status_code == 200
    assert init_response.json()["status"] == "initialized"
    assert init_response.json()["hublink_available"] is True


def test_node_neighbors_returns_graph_fragment(
    app, client, example_nodes, fake_service_factory, graph_explore_dependency
):
    """GET /api/v1/graph/node-neighbors should return the node list and record the nodeId argument."""
    fake_service = fake_service_factory(
        result=SimpleNamespace(
            answer="unused",
            nodes=example_nodes,
            sources=[],
            guardrails_warning=None,
        )
    )
    app.dependency_overrides[graph_explore_dependency] = lambda: fake_service

    response = client.get(
        "/api/v1/graph/node-neighbors",
        params={
            "nodeId": "R1",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["nodeId"] == "R1"
    assert len(payload["nodes"]) == 3
    assert payload["nodes"][0]["connections"][0]["targetId"] == "R2"
    assert fake_service.neighbor_calls == [{"node_id": "R1"}]
