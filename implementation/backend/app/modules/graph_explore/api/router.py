"""
Graph exploration API endpoints.
Handles one-hop knowledge graph neighbor expansion.
"""
import asyncio

from fastapi import APIRouter, HTTPException, Depends, Query
import logging

from backend.app.contracts.schemas import ErrorResponse, NodeNeighborsResponse
from backend.app.core.dependencies import get_graph_explore_service
from backend.app.modules.graph_explore.application.graph_explore_service import (
    GraphExploreService,
    GraphExploreUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get(
    "/node-neighbors",
    response_model=NodeNeighborsResponse,
    responses={
        200: {"description": "Successful response with graph neighbors"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        503: {"model": ErrorResponse, "description": "Graph explorer unavailable"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Expand a graph node by returning its direct neighbors",
)
async def get_node_neighbors(
    node_id: str = Query(..., alias="nodeId", min_length=1),
    explore_svc: GraphExploreService = Depends(get_graph_explore_service),
) -> NodeNeighborsResponse:
    """Return the direct-neighbor graph fragment for the requested node."""
    try:
        nodes = await asyncio.to_thread(explore_svc.get_node_neighbors, node_id=node_id)
        return NodeNeighborsResponse(node_id=node_id, nodes=nodes)
    except GraphExploreUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail="Graph explorer is currently unavailable."
        ) from e
    except Exception as e:
        logger.error("Error expanding node neighbors for node_id=%s: %s", node_id, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while expanding node '{node_id}'"
        ) from e
