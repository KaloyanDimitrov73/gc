"""Stats API router — secured endpoint for viewing request statistics."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config.base_settings import get_settings
from backend.app.modules.conversations.infrastructure.database.db import get_db
from backend.app.modules.stats.infrastructure.repository import StatsRepository

router = APIRouter(prefix="/stats", tags=["Stats"])
settings = get_settings()


def _verify_stats_key(x_stats_key: Annotated[str | None, Header()] = None) -> None:
    """Dependency: validate the X-Stats-Key header."""
    if not settings.stats_api_key:
        raise HTTPException(status_code=503, detail="Stats endpoint is not configured (STATS_API_KEY not set).")
    if x_stats_key != settings.stats_api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Stats-Key header.")


@router.get(
    "",
    summary="Request statistics",
    description=(
        "Returns aggregate and raw request timing statistics. "
        "Requires a valid `X-Stats-Key` header matching the `STATS_API_KEY` environment variable."
    ),
    dependencies=[Depends(_verify_stats_key)],
)
async def get_stats(
    hours: Annotated[int, Query(ge=1, le=8760, description="Time window in hours (default 24)")] = 24,
    limit: Annotated[int, Query(ge=1, le=500, description="Max number of recent requests to return (default 50)")] = 50,
    db: AsyncSession = Depends(get_db),
):
    repo = StatsRepository(db)
    return await repo.get_stats(hours=hours, limit=limit)
