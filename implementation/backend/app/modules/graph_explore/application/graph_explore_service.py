"""
Service for exploring the ORKG knowledge graph via direct-neighbor expansion.
"""
from typing import Any, List
import logging

from backend.app.contracts.schemas import GraphNode
from backend.app.modules.graph_explore.infrastructure.orkg.graph_explorer import GraphExplorer
from backend.app.shared.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


class GraphExploreUnavailableError(ServiceUnavailableError):
    """Raised when the graph explorer is not available."""


class GraphExploreService:
    """
    Owns the GraphExplorer and exposes direct-neighbor expansion for the API layer.
    """

    def __init__(self, graph: Any):
        self._explorer = GraphExplorer(graph) if graph is not None else None

    def get_node_neighbors(self, *, node_id: str) -> List[GraphNode]:
        if self._explorer is None:
            raise GraphExploreUnavailableError(
                "Graph explorer not initialized. Check graph loading logs."
            )
        return self._explorer.get_neighbor_nodes(node_id=node_id)

    def is_available(self) -> bool:
        return self._explorer is not None
