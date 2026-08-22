import os
import pickle
from typing import Any, Dict, Optional

from core.data.file_path_manager import FilePathManager
from core.logging.logging import get_logger

logger = get_logger(__name__)

BM25_INDEX_CACHE_NAME = "hublink_bm25_index"
SPLADE_INDEX_CACHE_NAME = "hublink_splade_index"


def bm25_index_path(index_key: str) -> str:
    """Returns the directory for a natively persisted BM25S index."""
    _validate_index_key(index_key)
    file_path_manager = FilePathManager()
    index_dir = file_path_manager.get_cache_path(BM25_INDEX_CACHE_NAME)
    return file_path_manager.combine_paths(index_dir, index_key, "path")


def splade_index_path(index_key: str) -> str:
    """Returns the persisted SPLADE index path for an index key."""
    _validate_index_key(index_key)
    return _index_path(SPLADE_INDEX_CACHE_NAME, f"{index_key}_path")


def save_sparse_index(file_path: str, data: Dict[str, Any]) -> None:
    """Persists a trusted sparse-index payload to disk."""
    FilePathManager().ensure_dir_exists(file_path)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_sparse_index(file_path: str) -> Optional[Dict[str, Any]]:
    """Loads a trusted sparse-index payload, returning None when unavailable."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
    except Exception as error:
        logger.error("Failed to load sparse index from %s: %s", file_path, error)
        return None
    if not isinstance(data, dict):
        logger.error("Sparse index at %s has an invalid payload", file_path)
        return None
    return data


def _index_path(cache_name: str, index_key: str) -> str:
    file_path_manager = FilePathManager()
    index_dir = file_path_manager.get_cache_path(cache_name)
    return file_path_manager.combine_paths(index_dir, f"{index_key}.pkl")


def _validate_index_key(index_key: str) -> None:
    if (not index_key
            or index_key in {".", ".."}
            or os.path.isabs(index_key)
            or os.path.basename(index_key) != index_key):
        raise ValueError("Sparse index key must be a non-empty file name.")
