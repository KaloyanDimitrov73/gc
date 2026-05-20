# Backend Structure

```text
backend/
├── app/                          # Application code (FastAPI app, modules, shared logic)
│   ├── core/                     # Cross-cutting runtime setup (config, DI, logging, registry)
│   ├── contracts/                # Shared request/response schemas used across modules
│   ├── shared/                   # Shared utilities and abstractions reused across modules
│   ├── modules/                  # Domain modules grouped by business capability (see below)
│   └── main.py                   # FastAPI application entry point
├── tests/                        # Backend-level integration and unit tests (see tests/README.md)
├── HublinkImplementation/        # HubLink submodule — retrieval pipeline and KG tooling
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── requirements.txt
├── init_graph_cache.py           # One-time setup: upload papers to ORKG and build local cache
└── docker-entrypoint.sh          # Container startup script
```

---

## Modules

```text
app/modules/
├── qa/            # Handles question-answering via the HubLink retrieval pipeline
├── conversations/ # Manages conversations (create, list, delete, history)
├── graph_load/    # Loads the ORKG knowledge graph into the local cache at startup (no API)
├── graph_explore/ # Graph traversal — returns neighbouring nodes for a given entity
└── guardrails/    # Validates inputs and outputs (gibberish, jailbreak, toxic language)
```

---

## Module Structure

Each module follows a three-layer layout:

```text
<module>/
├── api/                   # FastAPI router — HTTP endpoints, request/response models
├── application/           # Service class — business logic, no framework dependencies
├── infrastructure/        # External integrations (database, ORKG, HubLink, …)
│   └── <provider>/        # One subdirectory per external system
└── __init__.py
```

| Layer | Responsibility |
|---|---|
| `api/` | Defines HTTP routes and translates HTTP requests/responses. Depends on `application/`. |
| `application/` | Orchestrates business logic. Has no knowledge of HTTP or external systems. |
| `infrastructure/` | Carries out the actual work: database queries, ORKG REST calls, HubLink retrieval, cache I/O. Called by the application service. |

**Exceptions:**

- `graph_load` has no `api/` layer — it runs at startup, wired directly via dependency injection in `core/dependencies.py`.
- `guardrails` has no `api/` or `infrastructure/` layer — it contains only pure validation logic called internally by other modules.
- `qa` has an additional `configs/` directory for HubLink JSON configuration files.

---

## First-time Setup

Run `init_graph_cache.py` once before starting the backend for the first time:

```bash
python init_graph_cache.py
# or, to force re-uploading all papers:
python init_graph_cache.py --force-publication-update
```

**Requirements:** `ORKG_EMAIL` and `ORKG_PASSWORD` must be set in `.env`.

What it does:
1. Uploads papers from the local dataset (`merged_ecsa_icsa.json`) to the ORKG sandbox — skips papers that already exist.
2. Downloads the subgraph from ORKG and saves it as a local JSON cache.

After this runs once, the backend starts without credentials and serves from the local cache on every subsequent start. See [`app/modules/qa/README.md`](app/modules/qa/README.md) for flags that control cache refresh (`force_cache_update`, `force_index_update`, etc.).

---

## Notes

- `app/modules/*` is the main place for feature development.
- Router-exposed modules follow the `module/api/router.py` pattern and are wired in `core/module_registry.py`.
- `app/shared` should only contain reusable code with no module-specific assumptions.
- Keep tests close to the scope they validate: module tests under `modules/<name>/tests/`, cross-module tests under `backend/tests/`. See [`tests/README.md`](tests/README.md) for a detailed test report and explanation.
