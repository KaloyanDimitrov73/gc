"""
FastAPI main application.
Bridge between the React frontend and HubLink backend.
"""
# onnxruntime must be imported before torch to avoid DLL conflicts on Windows
import onnxruntime  # noqa: F401
import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.core.module_registry import api_v1_router
from backend.app.core.config import get_settings
from backend.app.core.dependencies import (
    get_graph_load_service,
    get_guardrails_service,
    get_graph_explore_service,
    get_hublink_service,
    get_retrieval_service,
)
from backend.app.modules.conversations.infrastructure.database.db import init_db, AsyncSessionLocal
from backend.app.modules.conversations.infrastructure.database.models import RequestLogModel
from backend.app.shared.exceptions import AppError

settings = get_settings()

# Configure logging
LOG_LEVEL = settings.log_level.upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if settings.suppress_healthcheck_access_log:
    logging.getLogger("uvicorn.access").addFilter(
        lambda r: '"GET /health HTTP/1.1" 200' not in r.getMessage()
    )

if settings.suppress_orkg_paged_false_warning:
    # ORKG uses loguru (not stdlib logging) for this warning.
    try:
        from loguru import logger as loguru_logger
        loguru_logger.disable("orkg.out")
    except Exception:
        # Keep startup robust even if loguru import/configuration changes.
        pass


# ---------------------------------------------------------------------------
# Request timing helpers
# ---------------------------------------------------------------------------

_SKIP_LOG_PATHS = {"/health", "/api/v1/qa/health", "/api/v1/qa/init", "/api/v1/stats"}

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _normalize_path(path: str) -> str:
    """Replace UUID path segments with ``{id}`` for clean aggregation."""
    return _UUID_RE.sub("{id}", path)


async def _save_request_log(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: str | None,
) -> None:
    """Persist a single request log entry. Runs as a fire-and-forget task."""
    try:
        async with AsyncSessionLocal() as session:
            entry = RequestLogModel(
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
                user_id=user_id,
            )
            session.add(entry)
            await session.commit()
    except Exception:
        logger.exception("Failed to persist request log entry")


class _TimedStreamWrapper:
    """Wraps a streaming response body to measure total stream duration."""

    def __init__(self, body_iterator, start: float, method: str, path: str, status_code: int, user_id: str | None):
        self._iterator = body_iterator
        self._start = start
        self._method = method
        self._path = path
        self._status_code = status_code
        self._user_id = user_id

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._iterator.__anext__()
            return chunk
        except StopAsyncIteration:
            duration_ms = (time.monotonic() - self._start) * 1000
            asyncio.create_task(
                _save_request_log(self._method, self._path, self._status_code, duration_ms, self._user_id)
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting HubLink FastAPI Bridge...")
    await init_db()
    logger.info("Database initialized.")

    async def _init_services() -> None:
        try:
            logger.info("Initializing services in background...")
            # Phase A: independent services in parallel
            await asyncio.gather(
                asyncio.to_thread(get_graph_load_service),
                asyncio.to_thread(get_guardrails_service),
            )
            logger.info("Graph and guardrails services initialized.")
            # Phase B: services that depend on graph_load, in parallel
            await asyncio.gather(
                asyncio.to_thread(get_graph_explore_service),
                asyncio.to_thread(get_hublink_service),
            )
            logger.info("Graph explore and HubLink services initialized.")
            # Phase C: retrieval service depends on hublink + guardrails
            await asyncio.to_thread(get_retrieval_service)
            logger.info("All services initialized and ready.")
        except Exception:
            logger.exception("Background service initialization failed")

    _init_task = asyncio.create_task(_init_services())
    yield
    # Shutdown
    _init_task.cancel()
    logger.info("Shutting down HubLink FastAPI Bridge...")


# Create FastAPI application
app = FastAPI(
    title="HubLink FastAPI Bridge",
    description="""
    FastAPI bridge between the React frontend and HubLink backend.

    This API provides:
    - Question answering using HubLink retrieval from ORKG
    - Knowledge graph visualization data
    - Support for multiple retrieval strategies (direct/graph traversal)
    - LLM-based answer generation
    - Automatic fallback to mock data when HubLink is unavailable
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.expose_docs else None,
    redoc_url="/redoc" if settings.expose_docs else None,
    openapi_url="/openapi.json" if settings.expose_docs else None,
)

# Include routers
app.include_router(api_v1_router)

@app.middleware("http")
async def check_request_size(request: Request, call_next):
    """Reject requests whose ``Content-Length`` exceeds ``max_request_body_bytes``.

    Returns HTTP 413 if the payload is too large, or HTTP 400 if the
    ``Content-Length`` header is present but not a valid integer.
    """
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_request_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request payload too large"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid Content-Length header"},
            )

    return await call_next(request)


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    """Measure and persist request duration for all non-health endpoints.

    For regular responses the duration is recorded after ``call_next`` returns.
    For streaming responses (SSE) the body iterator is wrapped so the duration
    is measured until the last chunk is consumed by the client.
    """
    if request.url.path in _SKIP_LOG_PATHS:
        return await call_next(request)

    method = request.method
    path = _normalize_path(request.url.path)
    user_id: str | None = request.cookies.get("user_id")
    start = time.monotonic()

    response = await call_next(request)

    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        response.body_iterator = _TimedStreamWrapper(
            response.body_iterator,
            start=start,
            method=method,
            path=path,
            status_code=response.status_code,
            user_id=user_id,
        )
    else:
        duration_ms = (time.monotonic() - start) * 1000
        asyncio.create_task(
            _save_request_log(method, path, response.status_code, duration_ms, user_id)
        )

    return response


@app.get("/health", tags=["Health"])
async def health():
    """
    Basic health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "HubLink FastAPI Bridge"
    }


# Global exception handlers
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """
    Catch-all for application errors that were not handled by a router.
    Routers translate known exceptions before they reach here, so this is a safety net.
    """
    logger.error("Unhandled application error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An application error occurred."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unhandled errors.
    """
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


def main():
    """CLI entry point for running the backend server."""
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
