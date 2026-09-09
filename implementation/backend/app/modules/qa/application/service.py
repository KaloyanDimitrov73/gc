"""
Orchestration service for retrieval and guardrails workflows.
"""
import logging
import threading
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Dict, Any

import asyncio as _asyncio

from backend.app.modules.guardrails.application.service import GuardrailsService
from backend.app.modules.qa.infrastructure.hublink.hublink_service import (
    HubLinkService,
)
from backend.app.contracts.schemas import GraphNode, MessageSchema
from backend.app.shared.exceptions import AppError, InputRejectedError, ServiceUnavailableError
from hublink.retrieval.utils.meta_router import RouteDecision

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

    def __init__(self, hublink_service: HubLinkService, guardrails_service: Optional[GuardrailsService]):
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
        if self._guardrails_service:
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

        guardrails_warning = None
        if self._guardrails_service:
            answer, validation_passed, warning = await self._guardrails_service.validate_output(answer)
            guardrails_warning = None if validation_passed else warning

        return RetrievalResult(
            answer=answer,
            nodes=nodes,
            sources=sources,
            guardrails_warning=guardrails_warning,
        )

    async def ask_streaming_with_meta_router(
            self,
            *,
            question: str,
            retrieval_mode: str,
            llm_model: str,
            number_of_hubs: int,
            topic_entity_id: Optional[str] = None,
            use_direct_final_answer: bool = False,
            conversation_history: Optional[List["MessageSchema"]],
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator that yields pipeline progress events and a final complete event.

        Progress events: {"type": "progress", "step": str, "percent": int}
        Complete event:  {"type": "complete", "answer": str, "nodes": list,
                          "sources": list, "guardrails_warning": str | None}
        Error events:    {"type": "error", "code": str, "detail": str}
        """
        yield {"type": "progress", "step": "input_validation", "percent": 5}

        if self._guardrails_service:
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

        history_text = _build_history_text(conversation_history)

        # --- Step 1: ask the meta router which path applies. This single call
        # replaces the old "always run general + chat_history in parallel"
        # approach -- it also directly returns the instant answer if the route
        # is GENERAL or CHAT_HISTORY, so there's no second round trip. ---
        yield {"type": "progress", "step": "routing_question", "percent": 10,
               "hubCompleted": None, "hubTotal": None}

        route, instant_answer, sources = await _asyncio.to_thread(
            self._hublink_service.get_routed_response, question, history_text
        )

        # --- Step 2a: route resolved to an instant answer (general / chat_history) ---
        if route != RouteDecision.RETRIEVAL and instant_answer is not None and instant_answer != "null":
            yield {"type": "progress", "step": "checking_instant_response", "percent": 20,
                   "hubCompleted": None, "hubTotal": None}

            valid_response, nodes = self._validate_instant_response(sources, conversation_history)
            if valid_response:
                yield {
                    "type": "complete",
                    "answer": instant_answer,
                    "nodes": nodes,
                    "sources": sources or [],
                    "guardrails_warning": None,
                }
                return

        # --- Step 2b: route is RETRIEVAL
        progress_queue: _asyncio.Queue = _asyncio.Queue()
        cancel_event = threading.Event()

        query_task = _asyncio.create_task(
            self._hublink_service.query_streaming(
                question=question,
                retrieval_mode=retrieval_mode,
                llm_model=llm_model,
                number_of_hubs=number_of_hubs,
                topic_entity_id=topic_entity_id,
                use_direct_final_answer=use_direct_final_answer,
                progress_queue=progress_queue,
                conversation_history=history_text,
                cancel_event=cancel_event,
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

        pending = {query_task}
        while pending:
            done, pending = await _asyncio.wait(
                pending, timeout=0.1, return_when=_asyncio.FIRST_COMPLETED
            )

            while not progress_queue.empty():
                evt = progress_queue.get_nowait()
                if evt and evt.get("type") == "hub_progress":
                    yield _queue_to_progress(evt)

        # Final drain -- events that arrived in the last tick
        while not progress_queue.empty():
            evt = progress_queue.get_nowait()
            if evt and evt.get("type") == "hub_progress":
                yield _queue_to_progress(evt)

        # NOTE: _finish_via_query_task previously also took an `instant_task` to
        async for evt in self._finish_via_query_task(query_task, None):
            yield evt

    async def ask_streaming(
        self,
        *,
        question: str,
        retrieval_mode: str,
        llm_model: str,
        number_of_hubs: int,
        topic_entity_id: Optional[str] = None,
        use_direct_final_answer: bool = False,
        conversation_history: Optional[List[MessageSchema]],
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator that yields pipeline progress events and a final complete event.

        Progress events: {"type": "progress", "step": str, "percent": int}
        Complete event:  {"type": "complete", "answer": str, "nodes": list,
                          "sources": list, "guardrails_warning": str | None}
        Error events:    {"type": "error", "code": str, "detail": str}
        """
        yield {"type": "progress", "step": "input_validation", "percent": 5}

        if self._guardrails_service:
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

        yield {"type": "progress", "step": "checking_instant_response", "percent": 10,
               "hubCompleted": None, "hubTotal": None}


        progress_queue: _asyncio.Queue = _asyncio.Queue()
        cancel_event = threading.Event()

        history_text = _build_history_text(conversation_history)

        instant_task = _asyncio.create_task(
            _asyncio.to_thread(
                self._hublink_service.get_instant_response, question, history_text
            )
        )

        query_task = _asyncio.create_task(
            self._hublink_service.query_streaming(
                question=question,
                retrieval_mode=retrieval_mode,
                llm_model=llm_model,
                number_of_hubs=number_of_hubs,
                topic_entity_id=topic_entity_id,
                use_direct_final_answer=use_direct_final_answer,
                progress_queue=progress_queue,
                conversation_history=history_text,
                cancel_event=cancel_event,
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
        instant_answer: Optional[str] = None
        instant_checked = False

        pending = {query_task, instant_task}

        while pending:
            done, pending = await _asyncio.wait(
                pending, timeout=0.1, return_when=_asyncio.FIRST_COMPLETED
            )

            while not progress_queue.empty():
                evt = progress_queue.get_nowait()
                if evt and evt.get("type") == "hub_progress":
                    yield _queue_to_progress(evt)

            if instant_task in done and not instant_checked:
                instant_checked = True
                instant_answer, sources = instant_task.result()

                if instant_answer is not None and instant_answer != "null":
                    valid_response, nodes = self._validate_instant_response(sources, conversation_history)
                    if valid_response:
                        async for evt in self._finish_via_instant_answer(
                                instant_answer, nodes, query_task, instant_task, cancel_event
                        ):
                            yield evt
                        return

            if query_task in done:
                break


        # Final drain — events that arrived in the last tick
        while not progress_queue.empty():
            evt = progress_queue.get_nowait()
            if evt and evt.get("type") == "hub_progress":
                yield _queue_to_progress(evt)

        async for evt in self._finish_via_query_task(query_task, instant_task):
            yield evt

    @staticmethod
    def _validate_instant_response(sources: Optional[List[int]], conversation_history: Optional[List[MessageSchema]]):
        nodes = []
        if sources:
            for i in sources:
                try:
                    idx = int(i)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid source index {i!r}: {e}")
                    return False, []

                if 0 <= idx < len(conversation_history):
                    message_nodes = conversation_history[idx].nodes
                    if not message_nodes:
                        continue
                    nodes.extend(message_nodes)
                else:
                    logger.warning(f"Source index out of range: {idx}")
                    return False, []

        return True, nodes

    async def _finish_via_instant_answer(
        self, instant_answer: str, nodes: List[GraphNode], query_task, instant_task, cancel_event: threading.Event
    ) -> AsyncIterator[Dict[str, Any]]:
        cancel_event.set()
        query_task.cancel()

        try:
            await query_task
        except _asyncio.CancelledError:
            pass

        yield {"type": "progress", "step": "instant_response", "percent": 90,
               "hubCompleted": None, "hubTotal": None}

        guardrails_warning = None
        if self._guardrails_service:
            instant_answer, validation_passed, warning = await self._guardrails_service.validate_output(instant_answer)
            guardrails_warning = None if validation_passed else warning

        yield {"type": "progress", "step": "saving", "percent": 95,
               "hubCompleted": None, "hubTotal": None}

        _assert_tasks_finished(query_task, instant_task, where="instant_answer_path")
        yield {"type": "complete", "answer": instant_answer, "nodes": nodes, "sources": [],
               "guardrails_warning": guardrails_warning, "contexts": None}

    async def _finish_via_query_task(self, query_task, instant_task) -> AsyncIterator[Dict[str, Any]]:
        try:
            answer, nodes, sources, contexts = await query_task
        except Exception as e:
            _assert_tasks_finished(query_task, instant_task, where="query_task_exception")
            print(e)
            yield {"type": "error", "code": "internal_error", "detail": str(e)}
            return

        yield {"type": "progress", "step": "processing", "percent": 93,
               "hubCompleted": None, "hubTotal": None}

        guardrails_warning = None
        if self._guardrails_service:
            answer, validation_passed, warning = await self._guardrails_service.validate_output(answer)
            guardrails_warning = None if validation_passed else warning

        yield {"type": "progress", "step": "saving", "percent": 95,
               "hubCompleted": None, "hubTotal": None}

        #_assert_tasks_finished(query_task, instant_task, where="retrieval_completion")
        yield {
            "type": "complete",
            "answer": answer,
            "nodes": [n.model_dump(by_alias=True) for n in nodes],
            "sources": sources,
            "guardrails_warning": guardrails_warning,
            "contexts": contexts
        }



    def get_status(self) -> Dict[str, Any]:
        """
        Return aggregated status for retrieval orchestration.
        """
        status = self._hublink_service.get_status()
        status["guardrails_available"] = self._guardrails_service.is_available()
        return status

def _assert_tasks_finished(*tasks: _asyncio.Task, where: str) -> None:
    unfinished = [t for t in tasks if not t.done()]
    if unfinished:
        logger.error(
            "ask_streaming exiting at '%s' with %d unfinished task(s): %s",
            where, len(unfinished), unfinished,
        )
    else:
        logger.info("ask_streaming exiting at '%s': all tasks finished.", where)

def _build_history_text(history: Optional[List[MessageSchema]]) -> str:
    if not history:
        return ""
    lines = []
    for i, turn in enumerate(history):
        line = f"[{i}] {turn.role}: {turn.content.strip()}"
        if turn.nodes:
            labels = [n.get("label", "") for n in turn.nodes if n.get("label")]
            if labels:
                line += f"\n    Related entities: {', '.join(labels)}"
        lines.append(line)
    return "\n".join(lines) + "\n"