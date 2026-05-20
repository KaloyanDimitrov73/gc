"""
Integration tests for QA router error handling.

Covers the four error branches of POST /api/v1/qa/ask:
- 422 when the service raises RetrievalInputRejectedError (guardrails block)
- 503 when the service raises RetrievalUnavailableError
- 500 for any unexpected exception (includes a request_id in the detail)
- 422 from Pydantic validation for an invalid retrievalMode or empty question

All real service implementations are replaced via app.dependency_overrides.
"""


def test_ask_returns_422_when_input_rejected(
    app, client, fake_service_factory, qa_router_module, retrieval_dependency
):
    """RetrievalInputRejectedError → 422 with a user-friendly rejection message."""
    fake_service = fake_service_factory(
        error=qa_router_module.RetrievalInputRejectedError("Rejected by guardrails")
    )
    app.dependency_overrides[retrieval_dependency] = lambda: fake_service

    response = client.post("/api/v1/qa/ask", json={"question": "blocked question"})

    assert response.status_code == 422
    assert response.json()["detail"] == "Your question was rejected by input safety checks."


def test_ask_returns_503_when_retrieval_unavailable(
    app, client, fake_service_factory, qa_router_module, retrieval_dependency
):
    """RetrievalUnavailableError → 503 indicating the backend service is down."""
    fake_service = fake_service_factory(
        error=qa_router_module.RetrievalUnavailableError("HubLink unavailable")
    )
    app.dependency_overrides[retrieval_dependency] = lambda: fake_service

    response = client.post("/api/v1/qa/ask", json={"question": "Any question"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Retrieval service is currently unavailable."


def test_ask_returns_500_when_unexpected_error_occurs(
    app, client, fake_service_factory, retrieval_dependency
):
    """Any unhandled exception → 500 with a detail string that contains a request_id for traceability."""
    fake_service = fake_service_factory(error=RuntimeError("Unexpected failure"))
    app.dependency_overrides[retrieval_dependency] = lambda: fake_service

    response = client.post("/api/v1/qa/ask", json={"question": "Any question"})

    assert response.status_code == 500
    assert "Internal server error (request_id=req-" in response.json()["detail"]


def test_ask_returns_422_for_invalid_retrieval_mode(
    app, client, fake_service_factory, retrieval_dependency
):
    """An unrecognised retrievalMode value should be caught by Pydantic before the service is called."""
    app.dependency_overrides[retrieval_dependency] = lambda: fake_service_factory(
        error=RuntimeError("Should not be called for invalid payload")
    )

    response = client.post(
        "/api/v1/qa/ask",
        json={
            "question": "Any question",
            "retrievalMode": "invalid-mode",
        },
    )

    assert response.status_code == 422
    detail_text = str(response.json()["detail"])
    assert "retrievalMode" in detail_text


def test_ask_returns_422_when_question_is_empty(
    app, client, fake_service_factory, retrieval_dependency
):
    """An empty question string should be rejected by Pydantic validation before reaching the service."""
    app.dependency_overrides[retrieval_dependency] = lambda: fake_service_factory(
        error=RuntimeError("Should not be called for invalid payload")
    )

    response = client.post("/api/v1/qa/ask", json={"question": ""})

    assert response.status_code == 422
    detail_text = str(response.json()["detail"])
    assert "question" in detail_text
