"""API router registry — assembles all module routers under the /api/v1 prefix."""
from fastapi import APIRouter
from backend.app.modules.qa.api.router import router as qa_router
from backend.app.modules.graph_explore.api.router import router as graph_explore_router
from backend.app.modules.conversations.api.router import router as conversations_router
from backend.app.modules.stats.api.router import router as stats_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(qa_router)
api_v1_router.include_router(graph_explore_router)
api_v1_router.include_router(conversations_router)
api_v1_router.include_router(stats_router)
