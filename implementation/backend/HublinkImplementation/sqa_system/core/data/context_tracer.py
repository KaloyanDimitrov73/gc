import json
from pydantic import BaseModel, Field

from sqa_system.core.logging.logging import get_logger
from sqa_system.core.data.file_path_manager import FilePathManager

logger = get_logger(__name__)


class ContextTrace(BaseModel):
    """
    A data object for tracing the creating of contexts back to the original text.
    """
    source_id: str = Field(..., description="The source id of the triple.")
    contexts: dict = Field(..., description="The contexts of the triples.")


class ContextTracer:
    """
    This tracing implementation allows for the tracing of contexts back to the original text.

    For example it is used during the creation of a knowledge graph to allow to trace back
    from which paper and exact location in the paper a triple was extracted.

    Args:
        context_id (str): The identifier that is used to identify the context in which the
            traces are stored. This is used to avoid collisions between different contexts.
    """

    def __init__(self, context_id: str):
        self.context_id = context_id
        self.context_trace_cache = self._load_context_trace_cache()

    def _load_context_trace_cache(self):
        """
        Loads a JSON file that contains the context traces.
        The file is located in the cache directory of the SQA system.
        """
        file_path_manager = FilePathManager()
        cache_path = file_path_manager.get_path("graph_trace_cache.json")
        if file_path_manager.file_path_exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}

    def _save_context_trace_cache(self):
        """
        Saves the context trace cache to a JSON file.
        """
        file_path_manager = FilePathManager()
        cache_path = file_path_manager.get_path("graph_trace_cache.json")
        file_path_manager.ensure_dir_exists(cache_path)
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(self.context_trace_cache, file, indent=2)

    def add_trace(self, trace_id: str, trace: ContextTrace):
        """
        Add a trace to the cache.

        Args:
            trace_id (str): The ID of the trace inside of the context.
            trace (ContextTrace): The trace to add.
        """
        self.context_trace_cache[f"{self.context_id}_{trace_id}"] = trace.model_dump(
        )
        self._save_context_trace_cache()

    def get_trace(self, trace_id: str) -> ContextTrace | None:
        """
        Get a trace from the cache.

        Args:
            trace_id (str): The ID of the trace.

        Returns:
            ContextTrace | None: The trace if it exists, None otherwise.
        """
        trace_data = self.context_trace_cache.get(
            f"{self.context_id}_{trace_id}",
            None
        )
        if trace_data is None:
            return None
        return ContextTrace.model_validate(trace_data)
