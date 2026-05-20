"""Stats repository — queries request_logs for aggregates and raw entries."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.modules.conversations.infrastructure.database.models import RequestLogModel


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Return the p-th percentile (0–100) of an already-sorted list."""
    if not sorted_values:
        return None
    idx = (p / 100) * (len(sorted_values) - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_values):
        return sorted_values[lower]
    fraction = idx - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


class StatsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_stats(self, hours: int, limit: int) -> dict:
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

        # --- Aggregate per endpoint ---
        agg_rows = (
            await self._session.execute(
                select(
                    RequestLogModel.method,
                    RequestLogModel.path,
                    func.count().label("count"),
                    func.sum(
                        (RequestLogModel.status_code >= 400).cast(type_=RequestLogModel.status_code.type)
                    ).label("error_count"),
                    func.avg(RequestLogModel.duration_ms).label("avg_duration_ms"),
                    func.min(RequestLogModel.duration_ms).label("min_duration_ms"),
                    func.max(RequestLogModel.duration_ms).label("max_duration_ms"),
                )
                .where(RequestLogModel.timestamp >= since)
                .group_by(RequestLogModel.method, RequestLogModel.path)
                .order_by(func.count().desc())
            )
        ).all()

        # --- Percentiles: fetch all durations per endpoint ---
        by_endpoint = []
        for row in agg_rows:
            durations_result = await self._session.execute(
                select(RequestLogModel.duration_ms)
                .where(
                    RequestLogModel.timestamp >= since,
                    RequestLogModel.method == row.method,
                    RequestLogModel.path == row.path,
                )
                .order_by(RequestLogModel.duration_ms)
            )
            durations = [r[0] for r in durations_result.all()]
            error_count = int(row.error_count or 0)
            by_endpoint.append(
                {
                    "method": row.method,
                    "path": row.path,
                    "count": row.count,
                    "error_count": error_count,
                    "avg_duration_ms": round(row.avg_duration_ms, 2) if row.avg_duration_ms is not None else None,
                    "p50_duration_ms": round(_percentile(durations, 50), 2) if durations else None,
                    "p95_duration_ms": round(_percentile(durations, 95), 2) if durations else None,
                    "min_duration_ms": round(row.min_duration_ms, 2) if row.min_duration_ms is not None else None,
                    "max_duration_ms": round(row.max_duration_ms, 2) if row.max_duration_ms is not None else None,
                }
            )

        # --- Summary ---
        total_requests = sum(r["count"] for r in by_endpoint)
        total_errors = sum(r["error_count"] for r in by_endpoint)
        all_durations_result = await self._session.execute(
            select(RequestLogModel.duration_ms)
            .where(RequestLogModel.timestamp >= since)
            .order_by(RequestLogModel.duration_ms)
        )
        all_durations = [r[0] for r in all_durations_result.all()]
        avg_all = round(sum(all_durations) / len(all_durations), 2) if all_durations else None

        summary = {
            "total_requests": total_requests,
            "error_count": total_errors,
            "avg_duration_ms": avg_all,
            "p50_duration_ms": round(_percentile(all_durations, 50), 2) if all_durations else None,
            "p95_duration_ms": round(_percentile(all_durations, 95), 2) if all_durations else None,
        }

        # --- Recent requests ---
        recent_rows = (
            await self._session.execute(
                select(
                    RequestLogModel.id,
                    RequestLogModel.timestamp,
                    RequestLogModel.method,
                    RequestLogModel.path,
                    RequestLogModel.status_code,
                    RequestLogModel.duration_ms,
                    RequestLogModel.user_id,
                )
                .where(RequestLogModel.timestamp >= since)
                .order_by(RequestLogModel.timestamp.desc())
                .limit(limit)
            )
        ).all()

        recent_requests = [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "method": r.method,
                "path": r.path,
                "status_code": r.status_code,
                "duration_ms": round(r.duration_ms, 2),
                "user_id": r.user_id,
            }
            for r in recent_rows
        ]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_hours": hours,
            "summary": summary,
            "by_endpoint": by_endpoint,
            "recent_requests": recent_requests,
        }
