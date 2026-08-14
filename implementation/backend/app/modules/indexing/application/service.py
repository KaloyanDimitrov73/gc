"""
Orchestration service for the indexing workflow.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


from backend.app.modules.graph_load.application.graph_load_service import GraphLoadService
from backend.app.modules.indexing.infrastructure.hublink.hub_indexing_service import (
    HubIndexingService,
)
from backend.app.modules.indexing.infrastructure.hublink.hub_store_service import (
    HubStoreService,
)
from backend.app.shared.exceptions import AppError

logger = logging.getLogger(__name__)


class IndexingServiceError(AppError):
    """Base exception for indexing orchestration failures."""


class IndexingAlreadyRunningError(IndexingServiceError):
    """Raised when a trigger is requested while a run is already in progress."""


@dataclass(frozen=True)
class IndexingRun:
    """Normalized result of a single indexing run."""

    run_id: str
    source: str  # "manual" | "scheduled"
    force: bool
    started_at: datetime
    finished_at: Optional[datetime]
    status: str  # "running" | "completed" | "failed"
    documents_indexed: Optional[int] = None
    error: Optional[str] = None


@dataclass
class _IndexingState:
    """Mutable in-memory state tracked between runs."""

    is_running: bool = False
    current_run_id: Optional[str] = None
    last_run: Optional[IndexingRun] = None
    history: List[IndexingRun] = field(default_factory=list)


class IndexingService:
    """
    Coordinates indexing runs, guarding against concurrent execution
    and exposing status for both the manual trigger and the background scheduler.

    Indexing itself runs in a dedicated single-thread executor so it never
    blocks the event loop (and therefore never blocks retrieval requests
    handled on the same loop).
    """

    def __init__(
        self,
        hub_indexing_service: HubIndexingService,
        vector_store_service: HubStoreService,
        graph_load_service: GraphLoadService,
    ):
        self._hub_indexing_service = hub_indexing_service
        self._vector_store_service = vector_store_service
        self._graph_load_service = graph_load_service
        self._lock = asyncio.Lock()
        self._state = _IndexingState()

        # Dedicated thread pool for indexing only.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexing")

        # Scheduler for continuous indexing
        self._scheduler_task: Optional[asyncio.Task] = None

    async def trigger_run(self, force: bool = False, source: str = "manual") -> IndexingRun:
        """
        Start an indexing run if none is currently in progress.

        Raises:
            IndexingAlreadyRunningError: if a run is already active.
        """
        if self._lock.locked():
            raise IndexingAlreadyRunningError("An indexing run is already in progress.")

        async with self._lock:
            run_id = f"idx-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            started_at = datetime.now(timezone.utc)
            self._state.is_running = True
            self._state.current_run_id = run_id

            logger.info("Starting indexing run run_id=%s source=%s force=%s", run_id, source, force)

            loop = asyncio.get_running_loop()

            try:
                entities_indexed = await loop.run_in_executor(
                    self._executor,
                    lambda: self._hub_indexing_service.run_indexing(
                        graph=self._graph_load_service.graph,
                        force_update=force,
                    ),
                )
                run = IndexingRun(
                    run_id=run_id,
                    source=source,
                    force=force,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    status="completed",
                    documents_indexed=entities_indexed,
                )
                logger.info(
                    "Indexing run completed run_id=%s documents_indexed=%s",
                    run_id,
                    entities_indexed,
                )
            except Exception as e:
                run = IndexingRun(
                    run_id=run_id,
                    source=source,
                    force=force,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    status="failed",
                    error=str(e),
                )
                logger.error("Indexing run failed run_id=%s: %s", run_id, e, exc_info=True)
            finally:
                self._state.is_running = False
                self._state.current_run_id = None
                self._state.last_run = run
                self._state.history.append(run)

            return run

    async def get_status(self, limit: int = 20) -> Dict[str, Any]:
        """
        Return current indexing state plus recent run history.
        """
        recent = self._state.history[-limit:][::-1]
        return {
            "is_running": self._state.is_running,
            "current_run_id": self._state.current_run_id,
            "last_run": self._state.last_run,
            "history": recent,
        }

    # ------------------------------------------------------------------ #
    # Automatic scheduling
    # ------------------------------------------------------------------ #

    def start_scheduler(self, interval_seconds: float) -> None:
        """
        Starts a background task that periodically calls `trigger_run`.
        Idempotent: calling it again won't start a second loop.
        """
        if self._scheduler_task is not None and not self._scheduler_task.done():
            logger.warning("Scheduler already running, ignoring start_scheduler() call.")
            return

        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(interval_seconds),
            name="indexing-scheduler",
        )
        logger.info("Indexing scheduler started, interval=%ss", interval_seconds)

    async def stop_scheduler(self) -> None:
        """
        Cleanly stops the scheduler task (e.g. on app shutdown).
        """
        if self._scheduler_task is None:
            return

        self._scheduler_task.cancel()
        try:
            await self._scheduler_task
        except asyncio.CancelledError:
            pass
        finally:
            self._scheduler_task = None
            logger.info("Indexing scheduler stopped.")

    async def _scheduler_loop(self, interval_seconds: float) -> None:
        """
        Runs indefinitely, triggering indexing runs at the given interval.
        Skips a cycle instead of raising if a (manual) run is already in progress.
        """
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.trigger_run(force=False, source="scheduled")
            except IndexingAlreadyRunningError:
                logger.info("Skipping scheduled indexing run: already running.")


