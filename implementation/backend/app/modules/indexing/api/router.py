"""Indexing API router — admin-secured endpoints to trigger and monitor indexing."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from backend.app.config.base_settings import get_settings
from backend.app.modules.indexing.application.service import (
    IndexingAlreadyRunningError,
    IndexingService,
)
from backend.app.core.dependencies import get_indexing_service

router = APIRouter(prefix="/indexing", tags=["Indexing"])
settings = get_settings()


def _verify_admin_key(x_admin_key: Annotated[str | None, Header()] = None) -> None:
    """Dependency: validate the X-Admin-Key header against the configured admin key."""
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=503,
            detail="Admin endpoints are not configured (ADMIN_API_KEY not set).",
        )
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Admin-Key header.")


@router.post(
    "/trigger_run",
    summary="Manually trigger an indexing run",
    description=(
        "Starts an indexing run immediately, outside the regular background schedule. "
        "Returns 409 if a run is already in progress. "
        "Requires a valid `X-Admin-Key` header matching the `ADMIN_API_KEY` environment variable."
    ),
    dependencies=[Depends(_verify_admin_key)],
)
async def trigger_indexing(
    force: Annotated[
        bool, Query(description="Force a full re-index instead of an incremental one")
    ] = False,
    indexing_svc: IndexingService = Depends(get_indexing_service),
):
    try:
        run = await indexing_svc.trigger_run(force=force, source="manual")
    except IndexingAlreadyRunningError as e:
        raise HTTPException(
            status_code=409,
            detail="An indexing run is already in progress.",
        ) from e
    return run


@router.get(
    "/status",
    summary="Indexing status",
    description=(
        "Returns the current indexing state plus recent run history. "
        "Requires a valid `X-Admin-Key` header matching the `ADMIN_API_KEY` environment variable."
    ),
    dependencies=[Depends(_verify_admin_key)],
)
async def get_indexing_status(
    limit: Annotated[
        int, Query(ge=1, le=500, description="Max number of recent runs to return (default 20)")
    ] = 20,
    indexing_svc: IndexingService = Depends(get_indexing_service),
):
    return await indexing_svc.get_status(limit=limit)