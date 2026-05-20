# Tests

```text
tests/
|- conftest.py                       # Root fixtures: --run-integration flag, integration skip logic
|- unit/                             # Fast, deterministic tests — no external services required
|  |- core/
|  |  `- test_config.py             # Validates default HubLink config file structure
|  `- modules/
|     |- graph_explore/
|     |  |- test_graph_explorer.py  # GraphExplorer neighbor traversal logic
|     |  `- test_node_builder.py    # Node depth computation and graph construction
|     `- graph_load/
|        `- test_graph_loader.py    # GraphLoader cache vs. full-init branching
|- integration/                      # Tests that require external services or runtime dependencies
|  |- api/
|  |  |- conftest.py                # TestClient fixtures, FakeRetrievalService, router wiring
|  |  |- test_qa_router.py          # QA and graph API endpoint behavior
|  |  `- test_qa_router_errors.py   # Error handling: 422, 503, 500 responses
|  `- modules/
|     |- qa/
|     |  |- test_embeddings.py      # Ollama and VDL embedding adapters
|     |  |- test_hublink_batch.py   # End-to-end HubLink query smoke test
|     |  `- test_llm.py             # Ollama, VDL, and OpenAI LLM adapters
|     `- guardrails/
|        |- test_service.py         # GuardrailsService input/output validation
|        `- results/                # Batch evaluation results per validator (CSV + notebook)
|           `- see_results.ipynb   # Analysis of validator performance (precision, recall, false negatives)
`- e2e/                              # Manual scripts and smoke tests for running backends
   |- results/                      # Saved batch retrieval outputs and neighbor fixtures
   |- helpers/
   |  `- vdl_model_checker.py       # Utility to list available models on the VDL server
   |- smoke_graph_gui.py            # Local GUI smoke test (mock backend + Vite frontend)
   |- capture_neighbor_fixtures.py  # Capture live node-neighbor responses for smoke fixtures
   `- test_hublink_batch.py         # Batch retrieval script over dataset rows
```

## Running tests

```bash
# All unit tests (fast, no external dependencies)
pytest tests/unit

# All integration tests (requires services and credentials — see .env.example)
pytest tests/integration --run-integration

# Everything
pytest --run-integration

# By marker
pytest -m unit
pytest -m integration --run-integration
```

Integration tests are **skipped by default**. Pass `--run-integration` to include them.

## Test types

| Type | Location | Speed | External deps |
|---|---|---|---|
| Unit | `tests/unit/` | Fast | None |
| Integration | `tests/integration/` | Slow | Ollama, VDL, ORKG, DB |
| E2E / scripts | `tests/e2e/` | Manual | Running backend |

## Notes

- `integration/api/` tests use a `FakeRetrievalService` and do not call real backends — they are fast but live under `integration/` because they spin up a full FastAPI router.
- `e2e/` scripts are not collected by pytest (no test functions). Run them directly with `python`.
- Integration tests that need credentials read from `.env` via `python-dotenv`. Copy `.env.example` to `.env` and fill in the required values before running.

### Gibberish validator — not in use

The `GibberishText` guardrails validator is **excluded from the default service configuration**. Batch evaluation showed an unacceptable false-positive rate: legitimate domain-specific queries are incorrectly flagged as gibberish, causing valid user inputs to be rejected.

See [`tests/integration/modules/guardrails/results/see_results.ipynb`](integration/modules/guardrails/results/see_results.ipynb).
