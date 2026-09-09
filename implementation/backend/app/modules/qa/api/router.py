"""
Question-Answering API endpoints.
Handles user questions and returns answers with knowledge graph visualization.
"""
import asyncio
import json
import threading

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import logging
from datetime import datetime
import uuid
from typing import AsyncIterator, Optional, List

from backend.app.config.base_settings import get_settings
from backend.app.contracts.schemas import (
    QuestionRequest,
    AnswerResponse,
    ErrorResponse, MessageSchema,
)
from backend.app.core.dependencies import get_retrieval_service, get_retrieval_service_if_ready, \
    get_conversation_service, get_llm_config_registry, get_or_create_user_id
from backend.app.modules.qa.application.service import (
    RetrievalService,
    RetrievalInputRejectedError,
    RetrievalUnavailableError,
)
from backend.app.modules.conversations.application.service import ConversationService
from backend.app.config.llm.llm_config_registry import LLMConfigRegistry
from backend.tests.integration.modules.qa.conftest import hublink_service
from hublink.retrieval.utils import answer_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qa", tags=["Question Answering"])


@router.get(
    "/llm-models",
    summary="List available LLM models",
    description="Returns all LLM models registered in llm_configs.json."
)
async def list_llm_models(
    registry: LLMConfigRegistry = Depends(get_llm_config_registry),
):
    """Return the list of configured LLM models with display metadata."""
    _provider_display = {
        "OpenAI": "OpenAI",
        "VDL": "Local (VDL)",
    }
    return [
        {
            "model": cfg.name_model,
            "provider": _provider_display.get(cfg.endpoint, cfg.endpoint),
        }
        for cfg in registry.list_models()
    ]


@router.post(
    "/init",
    summary="Initialize the HubLink service",
    description="Triggers lazy initialization of the HubLink service. "
                "Called by the frontend on startup to warm up the backend."
)
async def init_hublink(
    retrieval_svc: RetrievalService = Depends(get_retrieval_service)
):
    """Warm up the HubLink service so it's ready for queries."""
    status = retrieval_svc.get_status()
    return {
        "status": "initialized",
        "hublink_available": status["available"],
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post(
    "/ask",
    response_model=AnswerResponse,
    responses={
        200: {"description": "Successful response with answer and graph"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"model": ErrorResponse, "description": "Input rejected by safety checks"},
        503: {"model": ErrorResponse, "description": "Retrieval backend unavailable"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Ask a question about ORKG contributions",
    description="""
    Submit a question to be answered using the HubLink retrieval system.

    The system will:
    1. Parse the question and extract components
    2. Retrieve relevant contributions from ORKG based on the selected mode
    3. Generate an answer using the specified LLM
    4. Return the answer along with a knowledge graph for visualization

    """
)
async def ask_question(
    request: QuestionRequest,
    retrieval_svc: RetrievalService = Depends(get_retrieval_service),
    conversation_svc: ConversationService = Depends(get_conversation_service),
    user_id: str = Depends(get_or_create_user_id),
) -> AnswerResponse:
    """
    Process a user question and return an answer with knowledge graph.

    Args:
        request: Question request with query parameters
        retrieval_svc: Retrieval orchestration service

    Returns:
        AnswerResponse with answer text, graph nodes, and metadata
    """
    request_id = f"req-{uuid.uuid4()}"

    logger.info(
        "Received question request_id=%s mode=%s llm=%s hubs=%s question_len=%s conversation_id=%s",
        request_id,
        request.retrieval_mode,
        request.llm_model,
        request.number_of_hubs,
        len(request.question),
        request.conversation_id
    )

    conversation_history = None
    if request.conversation_id:
        conversation_history = await _get_conversation_history(
            request.conversation_id, user_id, conversation_svc
        )

    try:
        retrieval_result = await retrieval_svc.ask(
            question=request.question,
            retrieval_mode=request.retrieval_mode,
            llm_model=request.llm_model,
            number_of_hubs=request.number_of_hubs,
            topic_entity_id=request.topic_entity_id,
            use_direct_final_answer=request.use_direct_final_answer,
        )

        # Generate unique message ID
        message_id = f"msg-{uuid.uuid4()}"

        response = AnswerResponse(
            answer=retrieval_result.answer,
            nodes=retrieval_result.nodes,
            retrieval_mode=request.retrieval_mode,
            sources=retrieval_result.sources,
            message_id=message_id,
            guardrails_warning=retrieval_result.guardrails_warning,
        )

        logger.info(f"Successfully generated answer with {len(retrieval_result.nodes)} nodes")

        if request.conversation_id:
            try:
                await conversation_svc.persist_qa_exchange(
                    conversation_id=request.conversation_id,
                    question=request.question,
                    answer=retrieval_result.answer,
                    nodes=[n.model_dump(by_alias=True) for n in retrieval_result.nodes],
                    sources=retrieval_result.sources,
                )
            except Exception:
                logger.warning(
                    "Failed to persist messages for conversation %s — answer still returned",
                    request.conversation_id,
                )

        return response

    except RetrievalInputRejectedError as e:
        logger.info("Question rejected by guardrails request_id=%s", request_id)
        raise HTTPException(
            status_code=422,
            detail="Your question was rejected by input safety checks."
        ) from e
    except RetrievalUnavailableError as e:
        logger.warning("Retrieval unavailable request_id=%s reason=%s", request_id, str(e))
        raise HTTPException(
            status_code=503,
            detail="Retrieval service is currently unavailable."
        ) from e
    except Exception as e:
        logger.error("Error processing question request_id=%s: %s", request_id, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error (request_id={request_id})"
        )


@router.post(
    "/ask/stream",
    summary="Ask a question with streaming progress events",
    description="""
    Submit a question and receive Server-Sent Events (SSE) reporting
    pipeline progress, followed by the final answer as the last event.

    Event types emitted:
    - `progress`: `{"type": "progress", "step": str, "percent": int}`
    - `complete`: `{"type": "complete", "answer": str, "nodes": [...], "sources": [...], "messageId": str, "retrievalMode": str, "guardrailsWarning": str|null}`
    - `error`: `{"type": "error", "code": str, "detail": str}`
    """,
    response_class=StreamingResponse,
)

async def ask_question_stream(
    request: QuestionRequest,
    retrieval_svc: RetrievalService = Depends(get_retrieval_service),
    conversation_svc: ConversationService = Depends(get_conversation_service),
    user_id: str = Depends(get_or_create_user_id),
) -> StreamingResponse:
    """Stream pipeline progress events followed by the final answer."""
    request_id = f"req-{uuid.uuid4()}"
    logger.info(
        "Received streaming question request_id=%s mode=%s llm=%s hubs=%s question_len=%s conversation_id=%s",
        request_id,
        request.retrieval_mode,
        request.llm_model,
        request.number_of_hubs,
        len(request.question),
        request.conversation_id
    )

    conversation_history = None
    if request.conversation_id:
        conversation_history = await _get_conversation_history(
            request.conversation_id, user_id, conversation_svc
        )

    async def _sse_generator() -> AsyncIterator[str]:
        try:
            async for event in retrieval_svc.ask_streaming_with_meta_router(
                question=request.question,
                retrieval_mode=request.retrieval_mode,
                llm_model=request.llm_model,
                number_of_hubs=request.number_of_hubs,
                topic_entity_id=request.topic_entity_id,
                use_direct_final_answer=request.use_direct_final_answer,
                conversation_history=conversation_history
            ):
                if event.get("type") == "complete":
                    message_id = f"msg-{uuid.uuid4()}"

                    if request.conversation_id:
                        try:
                            await conversation_svc.persist_qa_exchange(
                                conversation_id=request.conversation_id,
                                question=request.question,
                                answer=event["answer"],
                                nodes=event["nodes"],
                                sources=event["sources"],
                            )
                        except Exception:
                            logger.warning(
                                "Failed to persist messages for conversation %s — answer still sent",
                                request.conversation_id,
                            )

                    payload = {
                        "type": "complete",
                        "answer": event["answer"],
                        "nodes": event["nodes"],
                        "sources": event["sources"],
                        "messageId": message_id,
                        "retrievalMode": request.retrieval_mode,
                        "guardrailsWarning": event.get("guardrails_warning"),
                    }

                    yield f"data: {json.dumps(payload)}\n\n"


                else:
                    yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error(
                "Unhandled error in streaming request_id=%s: %s", request_id, e, exc_info=True
            )
            error_payload = {
                "type": "error",
                "code": "internal_error",
                "detail": f"Internal server error (request_id={request_id})",
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/health",
    summary="Health check for QA service",
    description="Check if the QA service and HubLink backend are available"
)
async def health_check():
    """
    Health check endpoint. Returns immediately without triggering initialization.

    Returns:
        Status information about the service
    """
    retrieval_svc = get_retrieval_service_if_ready()
    if retrieval_svc is None:
        return {
            "status": "initializing",
            "hublinkAvailable": False,
            "hublink_available": False,
            "message": "Backend services are still initializing",
            "timestamp": datetime.utcnow().isoformat()
        }
    status = retrieval_svc.get_status()
    return {
        "status": "healthy",
        "hublinkAvailable": status["available"],
        "hublink_available": status["available"],
        "message": status["message"],
        "timestamp": datetime.utcnow().isoformat()
    }


async def _get_conversation_history(
    conversation_id: str,
    user_id: str,
    conversation_svc: ConversationService,
    settings = get_settings()
) -> Optional[List[MessageSchema]]:
    """Fetch the last 4 messages of a conversation, if it exists and belongs to the user."""
    try:
        conversation_details = await conversation_svc.get_conversation_detail(conversation_id, user_id)

        if conversation_details is None:
            logger.warning(
                "Conversation %s not found or not owned by user %s",
                conversation_id,
                user_id,
            )
            return None

        messages = conversation_details.messages[-settings.n_messages_history:]
        logger.info(
            "Get conversation history for conversation_id=%s with messages: \n%s",
            conversation_id,
            messages,
        )

        return messages

    except Exception:
        logger.warning(
            "Failed to get prior messages for conversation %s",
            conversation_id,
        )
        return None