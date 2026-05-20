# Implementation Overview

This directory contains the full-stack application: a React frontend, a FastAPI backend, and a Docker Compose setup that wires them together.

---

## Tech Stack

### Frontend

| | |
|---|---|
| **Framework** | React 18 |
| **Build tool** | Vite 6 (with SWC plugin for fast compilation) |
| **Styling** | Tailwind CSS 3 |
| **Components** | Radix UI (headless, accessible component primitives) |
| **Visualisation** | D3 7 + Recharts 2 (graph traversal views, charts) |
| **Forms** | react-hook-form |
| **Icons** | lucide-react |
| **Notifications** | sonner (toast messages) |
| **Layout** | react-resizable-panels |

The frontend runs as a **Vite dev server** (not a production build). API calls are proxied to the backend at `http://backend:8000`.

### Backend

| |                                                                                 |
|---|---------------------------------------------------------------------------------|
| **Framework** | FastAPI 0.115                                                                   |
| **Server** | uvicorn 0.32 (with `--reload` in Docker)                                        |
| **Runtime** | Python 3.12                                                                     |
| **Data validation** | Pydantic 2 + pydantic-settings                                                  |
| **Database** | SQLite via SQLAlchemy 2 + aiosqlite (conversation history)                      |
| **Safety validation** | Guardrails AI (jailbreak detection, toxic language, gibberish)                  |
| **Retrieval** | HubLink (submodule) — knowledge-graph retrieval pipeline over ORKG (RDF Graphs) |

See [`backend/README.md`](backend/README.md) for module structure and [`backend/app/modules/qa/README.md`](backend/app/modules/qa/README.md) for retrieval configuration.

---

## Docker

The application is orchestrated with `docker-compose.yml` at this directory level.

### Services

**`backend`**
- Built from `backend/Dockerfile` (Python 3.12-slim base image).
- Runs `uvicorn` with `--reload` — source changes inside the container are picked up automatically via the bind mount.
- Exposed internally on port **8000** (not published to the host directly).
- Health check: `GET http://localhost:8000/health` every 15 s, 30 s start period, 5 retries.

**`frontend`**
- Built from `frontend/Dockerfile` (Node 20 base image).
- Runs `npm ci && npm run dev` — installs dependencies on every start, then launches the Vite dev server.
- Published on **`localhost:3000`**.
- Proxies API requests to `http://backend:8000` via `VITE_DEV_PROXY_TARGET`.
- **Waits for the backend health check to pass** before starting.

### Volumes

| Volume | Type | Path in container | Purpose |
|---|---|---|---|
| `hublink_cache` | Named (Docker-managed) | `backend/HublinkImplementation/data/cache` | Persists the HubLink sqlite graph cache and ChromaDB vector store across restarts |
| `frontend_node_modules` | Named (Docker-managed) | `frontend/node_modules` | Caches npm packages so `npm ci` is fast on restart |
| `.:/app` | Bind mount | `/app` | Mounts the entire repo into both containers, enabling hot reload |

> **Note:** Named volumes survive `docker compose down` but are deleted by `docker compose down -v`. The bind mount ensures that the graph cache JSON files (written outside the named volume path) persist on the local filesystem.

### Guardrails Token

The backend requires a Guardrails token to configure its safety validators at runtime.

- Set `GUARDRAILS_TOKEN` in your shell environment or in a `.env` file at this directory level before running `docker compose up`.
- At container startup, `backend/docker-entrypoint.sh` runs `guardrails configure` with the token, then hands off to uvicorn.
- If the token is missing, a warning is printed and remote Guardrails inference will fail.

### Startup Sequence

```
docker compose up
  └── backend starts
        └── docker-entrypoint.sh: guardrails configure ...
        └── uvicorn starts on :8000
        └── health check passes (GET /health)
              └── frontend starts
                    └── npm ci (installs dependencies)
                    └── Vite dev server starts on :3000
```

### Common Commands

```bash
# Start everything
docker compose up -d

# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Stop (preserves volumes)
docker compose down

# Stop and delete all volumes (full reset)
docker compose down -v

# Rebuild images (e.g. after dependency changes)
docker compose up -d --build
```

---

## Request Statistics

The backend records the duration of every API request (excluding health checks) and exposes an aggregate statistics endpoint.

### Setup

Add the following to `backend/.env`:

```env
STATS_API_KEY=your-secret-key
```

The `request_logs` table is created automatically in the existing SQLite database on the next server start.

### Accessing the endpoint

```
GET /api/v1/stats
```

**Required header:** `X-Stats-Key: your-secret-key`

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `hours` | `24` | Time window to aggregate over (1–8760) |
| `limit` | `50` | Maximum number of raw log entries to return (1–500) |

**Example:**

```bash
curl -H "X-Stats-Key: your-secret-key" \
  "http://localhost:8000/api/v1/stats?hours=24&limit=50"
```

When running via Docker Compose, the backend is not published directly to the host. Use the frontend container's proxy or temporarily expose the backend port:

```bash
curl -H "X-Stats-Key: your-secret-key" \
  "http://localhost:3000/api/v1/stats?hours=24&limit=100"
```

### Response structure

```json
{
  "generated_at": "2026-04-23T12:00:00+00:00",
  "window_hours": 24,
  "summary": {
    "total_requests": 120,
    "error_count": 3,
    "avg_duration_ms": 842.5,
    "p50_duration_ms": 610.1,
    "p95_duration_ms": 3201.8
  },
  "by_endpoint": [
    {
      "method": "POST",
      "path": "/api/v1/qa/ask/stream",
      "count": 80,
      "error_count": 1,
      "avg_duration_ms": 1200.4,
      "p50_duration_ms": 980.2,
      "p95_duration_ms": 4500.0,
      "min_duration_ms": 310.5,
      "max_duration_ms": 6100.3
    }
  ],
  "recent_requests": [
    {
      "id": "uuid",
      "timestamp": "2026-04-23T11:59:00",
      "method": "POST",
      "path": "/api/v1/qa/ask/stream",
      "status_code": 200,
      "duration_ms": 980.2,
      "user_id": "uuid-or-null"
    }
  ]
}
```

> **Note:** Streaming requests (`/api/v1/qa/ask/stream`) are measured from the initial request until the last SSE event is consumed by the client, reflecting the full end-to-end duration.

