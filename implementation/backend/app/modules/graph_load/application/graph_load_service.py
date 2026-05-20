"""
Service for loading and caching the ORKG knowledge graph.
"""
from typing import Any, Optional
import logging

from backend.app.modules.graph_load.infrastructure.orkg.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


class GraphLoadService:
    """
    Owns graph loading and caching. Exposes the loaded graph instance
    for injection into HubLinkService and GraphExploreService.

    Graph loading behavior is controlled entirely by config:
    - force_cache_update=False (default): load from local JSON/sqlite cache
    - force_cache_update=True: re-download from ORKG sandbox on startup
    """

    def __init__(self, config: Any):
        self.graph: Optional[Any] = None
        self._available = False
        try:
            graph_loader = GraphLoader()
            self.graph = graph_loader.load_graph(config)
            self._available = True
            logger.info("GraphLoadService: graph loaded successfully")
        except Exception as e:
            logger.error("GraphLoadService: graph loading failed: %s", e, exc_info=True)

    def is_available(self) -> bool:
        return self._available
