"""
Orchestration service for retrieval and guardrails workflows.
"""
import logging
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Dict, Any

from backend.app.modules.guardrails.application.service import GuardrailsService
from backend.app.modules.qa.infrastructure.hublink.hublink_service import (
    HubLinkService,
)
from backend.app.contracts.schemas import GraphNode
from backend.app.shared.exceptions import AppError, InputRejectedError, ServiceUnavailableError

logger = logging.getLogger(__name__)

class RetrievalServiceError(AppError):
    """Base exception for retrieval orchestration failures."""


class RetrievalInputRejectedError(RetrievalServiceError, InputRejectedError):
    """Raised when input guardrails rejects the user query."""


class RetrievalUnavailableError(RetrievalServiceError, ServiceUnavailableError):
    """Raised when retrieval provider is unavailable."""


@dataclass(frozen=True)
class RetrievalResult:
    """Normalized response produced by retrieval orchestration."""

    answer: str
    nodes: List[GraphNode]
    sources: List[str]
    guardrails_warning: Optional[str] = None


class RetrievalService:
    """
    Coordinates guardrails and HubLink provider calls.
    """

    def __init__(self, hublink_service: HubLinkService, guardrails_service: GuardrailsService):
        self._hublink_service = hublink_service
        self._guardrails_service = guardrails_service

    async def ask(
        self,
        *,
        question: str,
        retrieval_mode: str,
        llm_model: str,
        number_of_hubs: int,
        topic_entity_id: Optional[str] = None,
        use_direct_final_answer: bool = False,
    ) -> RetrievalResult:
        """
        Validate input, query HubLink, and validate output.
        """
        is_valid, rejection_reason = await self._guardrails_service.validate_input(question)
        if not is_valid:
            raise RetrievalInputRejectedError(rejection_reason or "Input rejected by guardrails.")

        if not self._hublink_service.is_available():
            raise RetrievalUnavailableError(
                "HubLink service is not available. Try /api/v1/qa/init and check backend logs."
            )

        logger.info("Query Hublink.")

        answer, nodes, sources = await self._hublink_service.query(
            question=question,
            retrieval_mode=retrieval_mode,
            llm_model=llm_model,
            number_of_hubs=number_of_hubs,
            topic_entity_id=topic_entity_id,
            use_direct_final_answer=use_direct_final_answer,
        )

        answer, validation_passed, warning = await self._guardrails_service.validate_output(answer)
        return RetrievalResult(
            answer=answer,
            nodes=nodes,
            sources=sources,
            guardrails_warning=None if validation_passed else warning,
        )

    async def ask_streaming(
        self,
        *,
        question: str,
        retrieval_mode: str,
        llm_model: str,
        number_of_hubs: int,
        topic_entity_id: Optional[str] = None,
        use_direct_final_answer: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator that yields pipeline progress events and a final complete event.

        Progress events: {"type": "progress", "step": str, "percent": int}
        Complete event:  {"type": "complete", "answer": str, "nodes": list,
                          "sources": list, "guardrails_warning": str | None}
        Error events:    {"type": "error", "code": str, "detail": str}
        """
        yield {"type": "progress", "step": "input_validation", "percent": 10}

        is_valid, rejection_reason = await self._guardrails_service.validate_input(question)
        if not is_valid:
            yield {
                "type": "error",
                "code": "input_rejected",
                "detail": rejection_reason or "Input rejected by guardrails.",
            }
            return

        if not self._hublink_service.is_available():
            yield {
                "type": "error",
                "code": "unavailable",
                "detail": "HubLink service is not available. Try /api/v1/qa/init and check backend logs.",
            }
            return

        yield {"type": "progress", "step": "retrieving", "percent": 10,
               "hubCompleted": None, "hubTotal": None}

        import asyncio as _asyncio
        progress_queue: _asyncio.Queue = _asyncio.Queue()

        query_task = _asyncio.create_task(
            self._hublink_service.query_streaming(
                question=question,
                retrieval_mode=retrieval_mode,
                llm_model=llm_model,
                number_of_hubs=number_of_hubs,
                topic_entity_id=topic_entity_id,
                use_direct_final_answer=use_direct_final_answer,
                progress_queue=progress_queue,
            )
        )

        def _queue_to_progress(evt: Dict[str, Any]) -> Dict[str, Any]:
            step = evt.get("step", "retrieving")
            return {
                "type": "progress",
                "step": step,
                "percent": evt["percent"],
                "hubCompleted": evt.get("completed"),
                "hubTotal": evt.get("total"),
            }

        # Drain the progress queue while HubLink runs in its thread
        while not query_task.done():
            await _asyncio.sleep(0.1)
            while not progress_queue.empty():
                evt = progress_queue.get_nowait()
                if evt and evt.get("type") == "hub_progress":
                    yield _queue_to_progress(evt)

        # Final drain — events that arrived in the last tick
        while not progress_queue.empty():
            evt = progress_queue.get_nowait()
            if evt and evt.get("type") == "hub_progress":
                yield _queue_to_progress(evt)

        try:
            answer, nodes, sources = await query_task
        except Exception as e:
            yield {"type": "error", "code": "internal_error", "detail": str(e)}
            return

        yield {"type": "progress", "step": "processing", "percent": 93,
               "hubCompleted": None, "hubTotal": None}

        answer, validation_passed, warning = await self._guardrails_service.validate_output(answer)
        guardrails_warning = None if validation_passed else warning

        yield {"type": "progress", "step": "saving", "percent": 95,
               "hubCompleted": None, "hubTotal": None}

        yield {
            "type": "complete",
            "answer": answer,
            "nodes": [n.model_dump(by_alias=True) for n in nodes],
            "sources": sources,
            "guardrails_warning": guardrails_warning,
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Return aggregated status for retrieval orchestration.
        """
        status = self._hublink_service.get_status()
        status["guardrails_available"] = self._guardrails_service.is_available()
        return status
