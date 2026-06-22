"""
Service layer for interacting with the HubLink implementation.
Handles the integration between FastAPI and the HubLink retrieval system.
"""
from typing import Optional, List, Tuple, Dict, Any
import asyncio
import logging
import os
import threading

from backend.app.contracts.schemas import GraphNode
from backend.app.modules.qa.infrastructure.hublink.config_loader import (
    HublinkConfigLoader,
)
from backend.app.modules.qa.infrastructure.hublink.llm_config_registry import (
    LLMConfigRegistry,
)
from backend.app.modules.graph_explore.infrastructure.orkg import node_builder
from backend.app.modules.qa.infrastructure.hublink.setup_manager import (
    SetupManager,
)


logger = logging.getLogger(__name__)


class HubLinkService:
    """
    Service for interacting with the HubLink retrieval system.

    This service:
    - Receives an already-loaded knowledge graph from GraphLoadService
    - Queries the ORKG knowledge graph
    - Generates answers using LLMs
    - Transforms contexts to graph nodes for visualization
    """

    def __init__(self, graph: Any):
        """Initialize the HubLink service.

        Args:
            graph: Pre-loaded ORKG knowledge graph from GraphLoadService.
        """
        self.hublink_available = False
        self.retriever = None
        self.graph = graph
        self.config_loader = HublinkConfigLoader()
        self.llm_config_registry = LLMConfigRegistry()
        self.setup_manager = SetupManager()
        self._initialize_hublink()

    def _initialize_hublink(self):
        """
        Initialize the HubLink retriever using the pre-loaded graph.
        """
        try:
            if self.graph is None:
                raise RuntimeError("No knowledge graph provided to HubLinkService")

            # Import HubLink components
            from sqa_system.retrieval.implementations.HubLink.hub_link_retriever_for_user import HubLinkRetrieverForUser
            logger.info("HubLink modules imported successfully")

            # load default config for HubLink from JSON file
            config = self.config_loader.load_kg_retrieval_config()

            # Ensure API key is available in SecretManager for VDL-based configs.
            self.setup_manager.set_up_vdl_api_key()

            # Disable CLI progress bar - not needed in GUI context
            from core.progress.progress_handler import ProgressHandler
            ProgressHandler().disabled = False

            self.retriever = HubLinkRetrieverForUser(config, self.graph)

            # --- Override the LLM used for answer generation ---
            ANSWER_LLM_MODEL = os.getenv("ANSWER_LLM_MODEL")
            if ANSWER_LLM_MODEL:
                answer_llm_config = self.llm_config_registry.get(ANSWER_LLM_MODEL)
                if answer_llm_config is None:
                    logger.warning(
                        f"ANSWER_LLM_MODEL='{ANSWER_LLM_MODEL}' not found in llm_configs.json. "
                        f"Available: {list(self.llm_config_registry._configs_by_model.keys())}"
                    )
                else:
                    self.retriever._get_llm_adapter(answer_llm_config)
                    logger.info(f"Answer generation LLM overridden to: {ANSWER_LLM_MODEL}")

            self.hublink_available = True
            logger.info("HubLink service initialized successfully")
            self._setup_progress_streaming()

        except Exception as e:
            logger.error(f"HubLink initialization failed: {e}", exc_info=True)
            logger.warning("Falling back to mock mode")
            self.hublink_available = False

    def _setup_progress_streaming(self):
        """
        One-time patch of the ProgressHandler singleton to dispatch per-request
        streaming callbacks. Safe to call multiple times (no-op after first call).
        """
        try:
            from core.progress.progress_handler import ProgressHandler
            ph = ProgressHandler()

            if getattr(ph, '_streaming_callbacks_installed', False):
                return

            ph._streaming_callbacks = []
            ph._streaming_callbacks_lock = threading.Lock()
            ph._streaming_callbacks_installed = True

            _orig_add = ph.add_task
            _orig_update = ph.update_task_by_string_id
            _orig_finish = ph.finish_by_string_id

            def _dispatch_add(string_id, description, total, reset=False):
                result = _orig_add(string_id, description, total, reset)
                with ph._streaming_callbacks_lock:
                    for cb in list(ph._streaming_callbacks):
                        try:
                            cb('add', string_id, total)
                        except Exception:
                            pass
                return result

            def _dispatch_update(string_id, advance=1, remove_if_completed=True):
                result = _orig_update(string_id, advance, remove_if_completed)
                with ph._streaming_callbacks_lock:
                    for cb in list(ph._streaming_callbacks):
                        try:
                            cb('update', string_id, advance)
                        except Exception:
                            pass
                return result

            def _dispatch_finish(string_id):
                result = _orig_finish(string_id)
                with ph._streaming_callbacks_lock:
                    for cb in list(ph._streaming_callbacks):
                        try:
                            cb('finish', string_id, 0)
                        except Exception:
                            pass
                return result

            ph.add_task = _dispatch_add
            ph.update_task_by_string_id = _dispatch_update
            ph.finish_by_string_id = _dispatch_finish
            logger.info("ProgressHandler streaming callbacks installed")
        except Exception as e:
            logger.warning("Could not install ProgressHandler streaming hooks: %s", e)

    async def query(
        self,
        question: str,
        retrieval_mode: str,
        llm_model: str,
        number_of_hubs: int,
        topic_entity_id: Optional[str] = None,
        use_direct_final_answer: bool = False,
    ) -> Tuple[str, List[GraphNode], List[str]]:
        """
        Query the HubLink system with a question.

        Args:
            question: The user's question
            retrieval_mode: 'direct' or 'graph' retrieval strategy
            llm_model: LLM model identifier (e.g. 'llama3.1:8b', 'gpt-4o-mini')
            number_of_hubs: Number of hubs to retrieve
            topic_entity_id: Optional topic entity for graph traversal

        Returns:
            Tuple of (answer_text, graph_nodes, source_identifiers)

        Raises:
            RuntimeError: If HubLink is not initialized
        """
        logger.info("Query Hublink")

        if not self.hublink_available or self.retriever is None:
            logger.info("HubLink not available")
            raise RuntimeError("HubLink retriever not initialized")

        try:
            logger.info(
                "Querying HubLink mode=%s hubs=%s llm=%s question_len=%s",
                retrieval_mode,
                number_of_hubs,
                llm_model,
                len(question),
            )

            # Resolve the model name string to a full LLMConfig object
            llm_config = self.llm_config_registry.get(llm_model)
            if llm_config is None:
                raise ValueError(
                    f"Unknown LLM model '{llm_model}'. "
                    f"Available models: {list(self.llm_config_registry._configs_by_model.keys())}"
                )

            # Call the HubLink retriever in a thread so the async event loop
            # remains free to handle concurrent requests from other users.
            retrieval_answer = await asyncio.to_thread(
                self.retriever.query,
                query_text=question,
                number_of_hubs=number_of_hubs,
                retrieval_mode=retrieval_mode,
                llm_config=llm_config,
                topic_entity_id=topic_entity_id,
                use_direct_final_answer=use_direct_final_answer,
            )

            answer, nodes, sources = node_builder.build_answer_nodes_sources(retrieval_answer)

            logger.info(f"HubLink query completed: {len(nodes)} nodes, {len(sources)} sources")

            return answer, nodes, sources

        except Exception as e:
            logger.error(f"Error querying HubLink: {e}", exc_info=True)
            raise

    async def query_streaming(
        self,
        question: str,
        retrieval_mode: str,
        llm_model: str,
        number_of_hubs: int,
        topic_entity_id: Optional[str] = None,
        use_direct_final_answer: bool = False,
        progress_queue: Optional[asyncio.Queue] = None,
    ) -> Tuple[str, List[GraphNode], List[str]]:
        """
        Like query(), but pushes per-hub progress events to progress_queue while
        HubLink runs in a background thread.

        Queue events: {"type": "hub_progress", "completed": int, "total": int}
        """
        if not self.hublink_available or self.retriever is None:
            raise RuntimeError("HubLink retriever not initialized")

        try:
            from core.progress.progress_handler import ProgressHandler
            ph = ProgressHandler()
        except Exception:
            ph = None

        loop = asyncio.get_event_loop()

        # Per-task config: step name shown in UI + percent range
        _TASK_MAP = {
            "embedding_question": {
                "step": "retrieving",
                "pct_start": 12,
                "pct_end": 16,
                "show_hub_counter": False,
            },
            "candidate_hub_search": {
                "step": "retrieving",
                "pct_start": 16,
                "pct_end": 22,
                "show_hub_counter": False,
            },
            "path_filling": {
                "step": "analyzing",
                "pct_start": 22,
                "pct_end": 26,
                "show_hub_counter": False,
            },
            "hub_pruning": {
                "step": "analyzing",
                "pct_start": 26,
                "pct_end": 29,
                "show_hub_counter": False,
            },
            "partial_answer_generation": {
                "step": "analyzing",
                "pct_start": 30,
                "pct_end": 80,
                "show_hub_counter": False,
            },
            "final_answer_generation": {
                "step": "generating",
                "pct_start": 80,
                "pct_end": 99,
                "show_hub_counter": False,
            },
        }
        task_states: Dict[str, Dict[str, int]] = {}

        # Holds the ident of the worker thread running this specific request.
        # Populated inside _threaded_query() before the actual retrieval begins,
        # so that _progress_callback can reject events fired by other concurrent
        # requests whose threads are also registered in ph._streaming_callbacks.
        thread_ident_holder: list = []

        def _progress_callback(event_type: str, string_id: str, value: int):
            # Filter: only handle events from our own worker thread.
            if not thread_ident_holder:
                return
            if threading.current_thread().ident != thread_ident_holder[0]:
                return
            mapping = _TASK_MAP.get(string_id)
            if mapping is None or progress_queue is None:
                return
            if loop.is_closed():
                return
            state = task_states.setdefault(string_id, {'completed': 0, 'total': 1})
            if event_type == 'add':
                state['total'] = max(value, 1)
                state['completed'] = 0
                pct = mapping['pct_start']
            elif event_type == 'update':
                state['completed'] = min(state['completed'] + value, state['total'])
                span = mapping['pct_end'] - mapping['pct_start']
                pct = mapping['pct_start'] + int((state['completed'] / state['total']) * span)
            elif event_type == 'finish':
                state['completed'] = state['total']
                pct = mapping['pct_end']
            else:
                return
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {
                    "type": "hub_progress",
                    "step": mapping['step'],
                    "percent": pct,
                    "completed": state['completed'] if mapping.get('show_hub_counter') else None,
                    "total": state['total'] if mapping.get('show_hub_counter') else None,
                },
            )

        if ph is not None and getattr(ph, '_streaming_callbacks_installed', False):
            with ph._streaming_callbacks_lock:
                ph._streaming_callbacks.append(_progress_callback)

        try:
            llm_config = self.llm_config_registry.get(llm_model)
            if llm_config is None:
                raise ValueError(
                    f"Unknown LLM model '{llm_model}'. "
                    f"Available models: {list(self.llm_config_registry._configs_by_model.keys())}"
                )

            def _threaded_query():
                # Record this thread's identity so _progress_callback can filter
                # out events from other concurrent requests.
                thread_ident_holder.append(threading.current_thread().ident)
                return self.retriever.query(
                    query_text=question,
                    number_of_hubs=number_of_hubs,
                    retrieval_mode=retrieval_mode,
                    llm_config=llm_config,
                    topic_entity_id=topic_entity_id,
                    use_direct_final_answer=use_direct_final_answer,
                )

            retrieval_answer = await asyncio.to_thread(_threaded_query)
        finally:
            if ph is not None and getattr(ph, '_streaming_callbacks_installed', False):
                with ph._streaming_callbacks_lock:
                    try:
                        ph._streaming_callbacks.remove(_progress_callback)
                    except ValueError:
                        pass

        answer, nodes, sources = node_builder.build_answer_nodes_sources(retrieval_answer)
        logger.info("HubLink streaming query completed: %d nodes, %d sources", len(nodes), len(sources))
        return answer, nodes, sources

    def is_available(self) -> bool:
        """Check if HubLink is available."""
        return self.hublink_available

    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about the HubLink service.

        Returns:
            Dictionary with status information
        """
        status = {
            "available": self.hublink_available,
            "retriever_initialized": self.retriever is not None,
            "graph_initialized": self.graph is not None,
        }

        if self.hublink_available:
            status["ready"] = True
            status["message"] = "HubLink is ready"
        else:
            status["ready"] = False
            status["message"] = "HubLink not initialized. Check credentials and configuration."

        return status
