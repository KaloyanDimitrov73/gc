"""
Graph loading utilities for HubLink service initialization.
"""
from pathlib import Path
from typing import Any, Dict
import json
import logging

from core.data.file_path_manager import FilePathManager
from core.data.models import Triple, Knowledge
from knowledge_base.knowledge_graph.storage.implementations.orkg_remote_graph import ORKGRemoteGraph

logger = logging.getLogger(__name__)


class GraphLoader:
    """
    Encapsulates graph loading from local cache or ORKG sandbox.

    Reading from the ORKG REST API does not require credentials.
    Credentials are only needed when uploading papers (setup_graph_cache.py).

    Loading priority:
    1. If force_cache_update=True in config: re-download from ORKG unconditionally
    2. If sqlite cache already in memory: use it directly (fast path)
    3. If JSON cache file exists on disk: hydrate sqlite from it
    4. Otherwise (first time): download from ORKG and create the cache
    """

    def load_graph(self, config):
        """
        Load the ORKG knowledge graph.

        Respects force_cache_update from config.knowledge_graph_config.additional_params.
        """


        kg_config = config.knowledge_graph_config
        force_cache_update = kg_config.additional_params.get("force_cache_update", False)
        logger.info("KG_config: %s.", kg_config.config_hash)

        graph = ORKGRemoteGraph(kg_config)

        # Force re-download takes priority over any cached state
        if force_cache_update:
            logger.info("force_cache_update=True: re-downloading ORKG subgraph from sandbox.")
            graph.cache_subgraph()
            self._log_cache_metadata(kg_config)
            return graph

        # Fast path: sqlite cache already hydrated in memory
        existing_cache = graph.cache_manager.get_table(graph.cache_subgraph_key)
        if existing_cache:
            logger.info(
                "Using existing ORKG sqlite cache '%s' (%s entries).",
                graph.cache_subgraph_key,
                len(existing_cache),
            )
            self._log_cache_metadata(kg_config)
            return graph

        # Medium path: JSON snapshot exists on disk — hydrate sqlite from it
        cache_path = self._get_orkg_cache_path(kg_config)
        if Path(cache_path).exists():
            logger.info("Hydrating ORKG sqlite cache from file: %s", cache_path)
            self._hydrate_orkg_cache_tables_from_json(graph, cache_path)
            logger.info("Knowledge graph loaded from local ORKG cache successfully")
            self._log_cache_metadata(kg_config)
            return graph

        # First-time: no cache at all — download from ORKG and persist
        logger.info(
            "No local cache found. Downloading ORKG subgraph from sandbox (first-time setup). "
            "Consider running setup_graph_cache.py for future starts."
        )
        graph.cache_subgraph()
        self._log_cache_metadata(kg_config)
        return graph

    def _log_cache_metadata(self, kg_config):
        """Log when the graph cache was last built, if a metadata file exists.
        Falls back to the JSON file's modification time if no metadata file is found."""
        from datetime import datetime, timezone
        cache_path = self._get_orkg_cache_path(kg_config)
        meta_path = cache_path.replace(".json", ".meta.json")
        if Path(meta_path).exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                logger.info(
                    "Graph cache last built: %s | %s entries | source: %s",
                    meta.get("cached_at", "unknown"),
                    meta.get("entry_count", "unknown"),
                    meta.get("orkg_base_url", "unknown"),
                )
                return
            except Exception as e:
                logger.warning("Could not read graph cache metadata: %s", e)

        if Path(cache_path).exists():
            mtime = Path(cache_path).stat().st_mtime
            mtime_str = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            logger.info(
                "Graph cache last built: %s (estimated from file mtime — no metadata file found).",
                mtime_str,
            )
        else:
            logger.info("Graph cache metadata not found — cache age unknown.")

    def _get_orkg_cache_path(self, kg_config) -> str:
        """Return local ORKG JSON cache path for a KnowledgeGraphConfig."""


        fpm = FilePathManager()
        return fpm.combine_paths(
            fpm.KNOWLEDGE_GRAPH_DIR,
            "orkg",
            kg_config.config_hash + ".json"
        )

    @staticmethod
    def _is_triple_json(item: Dict[str, Any]) -> bool:
        return (
            "predicate" in item
            and "entity_subject" in item
            and "entity_object" in item
        )

    def _hydrate_orkg_cache_tables_from_json(self, graph, cache_path: str):
        """
        Load ORKG cached objects from JSON file into CacheManager sqlite tables.
        Uses batch inserts for performance.
        """

        with open(cache_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        if not isinstance(raw_items, list):
            raise ValueError(
                f"Unexpected cache format in '{cache_path}': expected a JSON list."
            )

        cache_manager = graph.cache_manager
        cache_manager.delete_table(graph.cache_subgraph_key)
        cache_manager.delete_table(graph.cache_publication_roots_key)

        subgraph_rows = []
        publication_rows = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            if self._is_triple_json(item):
                triple = Triple.model_validate(item)
                payload_json = triple.model_dump_json()
            else:
                knowledge = Knowledge.model_validate(item)
                payload_json = knowledge.model_dump_json()

                if graph.paper_type in (knowledge.knowledge_types or []):
                    # Double-serialize to match the storage format used by CacheManager.add_data(),
                    # which calls json.dumps() on the value before writing to SQLite.
                    # get_table() then does json.loads() on retrieval, yielding back the raw string.
                    publication_rows.append(
                        (cache_manager.get_hash_value(payload_json), json.dumps(payload_json))
                    )

            # Double-serialize to match add_data() format (see publication_rows comment above).
            subgraph_rows.append(
                (cache_manager.get_hash_value(payload_json), json.dumps(payload_json))
            )

        if not subgraph_rows:
            raise ValueError(f"Loaded 0 cache rows from '{cache_path}'.")

        cache_manager.add_data_batch(graph.cache_subgraph_key, subgraph_rows)
        cache_manager.add_data_batch(graph.cache_publication_roots_key, publication_rows)

        logger.info(
            "Hydrated ORKG cache from JSON: %s rows, %s publication roots",
            len(subgraph_rows),
            len(publication_rows),
        )
