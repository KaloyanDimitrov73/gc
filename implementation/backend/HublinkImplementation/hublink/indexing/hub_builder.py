from collections import deque
from typing import List, Tuple
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
from pydantic import BaseModel, Field, ConfigDict

from core.data.cache_manager import CacheManager
from core.data.models.triple import Triple
from core.progress.progress_handler import ProgressHandler
from hublink.core.models.entity_with_direction import EntityWithDirection
from hublink.core.models.hub import Hub, IsHubOptions
from hublink.core.models.hub_path import HubPath
from language_model.base.llm_adapter import LLMAdapter
from knowledge_base.knowledge_graph.storage.utils.graph_converter import GraphConverter
from knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from core.logging.logging import get_logger
from hublink.core.hub_storage_manager import HubStorageManager
from hublink.core.utils.hub_path_util import path_to_hash

logger = get_logger(__name__)


class HubBuilderOptions(BaseModel):
    """
    Configuration options for the HubBuilder, controlling hub construction and embedding.

    Args:
        is_hub_options (IsHubOptions): Options for classifying entities as hubs.
        max_workers (int): Maximum number of workers for parallel processing.
        llm (LLMAdapter): LLM used for converting graph structures to text.
        hub_storage_manager (HubStorageManager): Manager for vector store.
        max_hub_path_length (int): Maximum allowed length for a hub path.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    is_hub_options: IsHubOptions = Field(
        ...,
        title="Is Hub Options",
        description="The options used to classify an entity as a hub."
    )
    max_workers: int = Field(
        8,
        title="Max Workers",
        description="Maximum number of workers to use for parallel processing."
    )
    llm_workers: int = Field(
        100,
        title="LLM Workers",
        description=(
            "Maximum number of concurrent LLM calls (e.g. OpenAI API requests). "
            "Shared across all hub threads to prevent rate-limit errors. "
            "Lower this value if you hit 429 errors; raise it for local models."
        )
    )
    llm: LLMAdapter = Field(
        ...,
        title="LLM Adapter",
        description="The LLM that is used for inference."
    )
    hub_storage_manager: HubStorageManager = Field(
        ...,
        title="Hub Retrieval",
        description="Domain service for storing/retrieving HubPaths, built on top of vector_store.",
    )
    max_hub_path_length: int = Field(
        ...,
        title="Max Hub Path Length",
        description="The maximum length that a HubPath can have."
    )


class HubBuilder:
    """
    Builds hub structures in the knowledge graph for retrieval.

    The HubBuilder constructs hubs from validated root entities, embedding their paths and storing them in the vector store for efficient retrieval.

    Args:
        graph (KnowledgeGraph): The knowledge graph to build hubs from.
        options (HubBuilderOptions): Configuration for hub construction and model usage.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 options: HubBuilderOptions):
        self.graph = graph
        self.options = options
        self.progress_handler = ProgressHandler()
        self.processed_entities = set()
        self.visited_with_direction = set()
        self.cache_manager = CacheManager()
        self.graph_converter = GraphConverter(graph=self.graph,
                                              llm_adapter=self.options.llm)
        # Shared executor for LLM calls – created in build_hubs, shared across hub threads
        # so that global OpenAI concurrency never exceeds options.llm_workers.
        self._llm_executor: ThreadPoolExecutor | None = None

    def build_hubs(self,
                   hub_entities: List[EntityWithDirection],
                   update_cached_hubs: bool = False) -> Tuple[List[EntityWithDirection], List[Hub]]:
        """
        Assumes that the hub entities are already validated hubs.
        Each hub entity is the root of a hub. The building process starts from
        these roots to build the hubs.

        Args:
            hub_entities (List[EntitywithDirection]): The entities that are classified as hubs.
                These are the root entities of the hubs.
            update_cached_hubs (bool): If set to true, hubs that are already cached are checked
                for updates.

        Returns:
            Tuple[List[EntityWithDirection], List[Hub]]: The next traversal candidates
                and the hubs that were built.
        """
        if not hub_entities:
            return [], []

        hubs: List[Hub] = []
        next_traversal_candidates: List[EntityWithDirection] = []

        # One shared LLM executor caps total OpenAI concurrency to llm_workers,
        # regardless of how many hub threads run in parallel.
        with ThreadPoolExecutor(max_workers=self.options.llm_workers) as llm_executor:
            self._llm_executor = llm_executor
            try:
                with ThreadPoolExecutor(
                    max_workers=self.options.max_workers
                ) as executor:
                    futures = []

                    for hub_entity_with_direction in hub_entities:
                        future = executor.submit(
                            self._build_hub,
                            hub_entity_with_direction,
                            update_cached_hubs)
                        futures.append(future)

                    if futures:
                        self._process_futures(
                            futures=futures,
                            hubs=hubs,
                            next_traversal_candidates=next_traversal_candidates
                        )
            finally:
                self._llm_executor = None

        return next_traversal_candidates, hubs

    def _process_futures(self,
                         futures: List[Tuple[List[EntityWithDirection], Hub]],
                         hubs: List[Hub],
                         next_traversal_candidates: List[EntityWithDirection]):
        """
        Processes the results from the Hub processing futures.

        Args:
            futures (List[Tuple[List[EntitywithDirection], Hub]]): The futures to process.
            hubs (List[Hub]): The list of hubs to append the processed hubs to.
            next_traversal_candidates (List[EntitywithDirection]): The list of next traversal
                candidates which are the entities that are not part of the current hubs.
                These can be used in subsequent calls to continue the traversal.
        """
        futures_task = self.progress_handler.add_task(
            description="Processing Hubs...",
            total=len(futures),
            string_id="hub_processing"
        )

        total = len(futures)
        completed = 0
        logger.info("Indexing: starting processing of %d hubs", total)



        # Here we collect the future results from the hub processing
        for future in as_completed(futures):
            next_traversal_canidates, hub = future.result()
            next_traversal_candidates.extend(next_traversal_canidates)
            hubs.append(hub)
            completed += 1
            remaining = total - completed
            logger.info(
                "Indexing: hub %s done (%d/%d, %d remaining)",
                hub.root_entity.entity.uid,
                completed,
                total,
                remaining,
            )

            # Because a hub is only traversed in the forward direction of the graph
            # we need to add the left entities as candidates in case that we are
            # processing the graph to the "left" side (against the direction of the graph)
            is_left = hub.root_entity.left
            if is_left:
                relations = self.graph.get_relations_of_tail_entity(
                    hub.root_entity.entity)
                next_traversal_candidates.extend([
                    EntityWithDirection(
                        entity=relation.entity_subject,
                        left=False,  # Candidates from the hub are always right
                        path_from_topic=hub.root_entity.path_from_topic +
                        [relation]
                    ) for relation in relations])
            self.progress_handler.update_task_by_string_id(
                futures_task, advance=1)
        self.progress_handler.finish_by_string_id(futures_task)

    def _build_hub(self,
                   hub_root_entity: EntityWithDirection,
                   update_cached_hubs: bool) -> Tuple[List[EntityWithDirection], Hub]:
        """
        Processes a hub entity by retrieving and caching its associated paths.

        Args:
            hub_root_entity (EntitywithDirection): The root entity of the hub to process.
            update_cached_hubs (bool): If set to true, the hub is processed
                regardless of whether it has been cached before.

        Returns:
            Tuple[List[EntitywithDirection], Hub]: The next traversal candidates
                and the hub that was built.
        """

        logger.debug("Processing hub %s ...", hub_root_entity.entity.uid)

        # In case that we can use the cached hubs without needing to
        # check for updates, we get the cached data
        hub_paths = []
        if not update_cached_hubs:
            next_hub_roots, hub_paths = self.get_cached_hub_paths(
                hub_root_entity=hub_root_entity,
            )


        # If no cached data is found, we process the hub and save the data
        # into the cache
        if len(hub_paths) == 0:
            logger.debug(
                ("No cached data found for hub %s or hub update is set "
                 "to true. Processing hub.."),
                hub_root_entity.entity.uid)

            paths, next_hub_roots = self._get_hub_paths(
                hub_root_entity=hub_root_entity)

            needs_rebuilding, cached_paths = self._check_if_hub_needs_rebuilding(
                hub_root_entity=hub_root_entity,
                paths=paths
            )

            if needs_rebuilding:
                # Process the paths that have been found
                hub_paths = self._build_hub_paths(paths, hub_root_entity)
                logger.debug("Process path")
            else:
                # Reuse the paths already loaded by _check_if_hub_needs_rebuilding
                # – avoids a second full ChromaDB scan of the hub.
                hub_paths = cached_paths
                logger.debug("Reuse the paths")

            # save into the cache
            self.cache_manager.add_data(
                meta_key=f"hub_paths_{self.options.hub_storage_manager.vector_store_name()}",
                dict_key=hub_root_entity.entity.uid,
                value={
                    "next_hub_roots": next_hub_roots,
                    "hub_path_hash": [path.path_hash for path in hub_paths]
                }
            )
            logger.debug("Finished processing hub..")
            logger.debug("Hub paths: %d", len(hub_paths))

        processed_hub = Hub(
            root_entity=hub_root_entity,
            paths=hub_paths
        )
        return next_hub_roots, processed_hub

    def get_cached_hub_paths(self,
                             hub_root_entity: EntityWithDirection
                             ) -> Tuple[List[EntityWithDirection], List[HubPath]]:
        """
        Looks up whether the hub paths have been cached from previous
        processing.

        HubPaths have a unique hash value and are stored in the vector
        store by their hash ID with all the necessary metadata like its
        text description and the triples.

        To get the Hubpaths of a Hub, we store the HashValue of the
        path in a seperate cache. This cache is a dictionary with the
        unique hash of the Hub itself and a list of all the hash values
        of the HubPaths the hub contains.

        Args:
            hub_root_entity (EntitywithDirection): The root entity of the hub to process.

        Returns:
            Tuple[List[EntitywithDirection], List[HubPath]]: The next traversal candidates
                and the hub paths that were built.
        """
        # We look up the cache to find whether the paths of the hub
        # have been processed before. The lookup is done using the
        # hubs root entity uid.
        cached_data = self.cache_manager.get_data(
            meta_key=f"hub_paths_{self.options.hub_storage_manager.vector_store_name()}",
            dict_key=hub_root_entity.entity.uid,
        )

        # Return empty lists if no cached data is found
        if cached_data is None:
            return [], []

        # Even when paths are found, there are instances where the
        # paths are not in the vector store. In that case we need to
        # process all the paths again.
        all_paths_loaded = True
        logger.debug(
            "Cached data found for hub %s. Using cached data..",
            hub_root_entity.entity.uid)
        next_hub_roots = cached_data["next_hub_roots"]
        path_keys = cached_data["hub_path_hash"]

        # Fetch all paths in a single batch DB call instead of one call per path.
        retrieved = self.options.hub_storage_manager.retrieve_hub_paths_by_keys(path_keys)
        hub_paths = []
        for path_hash in path_keys:
            hub_path = retrieved.get(path_hash)
            if hub_path is not None:
                hub_paths.append(hub_path)
            else:
                all_paths_loaded = False
                break

        logger.debug("Amount of paths in cache: %d", len(hub_paths))
        if not all_paths_loaded:
            logger.warning(
                "Not all paths of hub %s are loaded from the cache. Processing hub..",
                hub_root_entity.entity.uid)
            return [], []
        return next_hub_roots, hub_paths

    def _get_hub_paths(self,
                       hub_root_entity: EntityWithDirection
                       ) -> Tuple[List[List[Triple]], List[EntityWithDirection]]:
        """
        Processes a Hub by its root entity by creating all its
        Hubpaths. The Hubpaths are created by traversing the graph
        from the root entity to the end or the next hub.

        Args:
            hub_root_entity (EntitywithDirection): The root entity of the hub to process.

        Returns:
            Tuple[List[List[Triple]], List[EntitywithDirection]]: The paths of the hub
                and the next traversal candidates.
        """
        paths: List[List[Triple]] = []
        visited = set()
        queue = deque([(hub_root_entity.entity, [])])
        next_hub_roots: List[EntityWithDirection] = []

        thread_id = str(threading.current_thread().ident)
        progress_task = self.progress_handler.add_task(
            string_id=f"processing_hub_paths_{thread_id}",
            description=f"Processing Hub Paths for {hub_root_entity.entity.uid}",
            total=len(queue)
        )

        while queue:
            current_entity, current_path = queue.popleft()
            logger.debug("Searching for paths.. Current entity %s",
                         current_entity.uid)

            if current_entity.uid in visited and current_entity.uid != hub_root_entity.entity.uid:
                self.progress_handler.update_task_by_string_id(
                    progress_task, advance=1)
                continue

            # Check if the maximum length of the path is reached
            if self.options.max_hub_path_length > 0 and len(current_path) >= self.options.max_hub_path_length:
                paths.append(current_path)
                self.progress_handler.update_task_by_string_id(
                    progress_task, advance=1)
                continue

            visited.add(current_entity.uid)

            # Only get outbound relations
            relations = self.graph.get_relations_of_head_entity(current_entity)

            # If no outbound relations, treat it as end node
            if not relations:
                # Only add if path exists
                if current_path and len(current_path) > 0:
                    paths.append(current_path)
                self.progress_handler.update_task_by_string_id(
                    progress_task, advance=1)
                continue

            # Process each outbound relation
            for relation in relations:
                next_entity = relation.entity_object

                # Skip if going back to hub
                if next_entity.uid == hub_root_entity.entity.uid:
                    self.progress_handler.update_task_by_string_id(
                        progress_task, advance=1)
                    continue

                # Check if next entity is a hub
                is_next_hub = Hub.is_hub_entity(
                    entity=next_entity,
                    graph=self.graph,
                    options=self.options.is_hub_options
                )

                if is_next_hub:
                    # Create hub path when reaching another hub
                    next_hub_roots.append(EntityWithDirection(
                        entity=next_entity,
                        left=False,
                        path_from_topic=hub_root_entity.path_from_topic +
                        current_path + [relation]
                    ))
                    if current_path and len(current_path) > 0:
                        paths.append(current_path)
                else:
                    # Continue exploring
                    queue.append((next_entity, current_path + [relation]))
                    self.progress_handler.update_task_length(
                        progress_task, len(queue))

        self.progress_handler.finish_by_string_id(progress_task)

        return paths, next_hub_roots

    def _check_if_hub_needs_rebuilding(self,
                                       paths: List[List[Triple]],
                                       hub_root_entity: EntityWithDirection) -> Tuple[bool, List[HubPath]]:
        """
        Evaluates whether the content of the hub has been updated.
        If so, it marks the hub as needing to be rebuilt.

        This is done by comparing the current paths of the hub with the
        cached paths in the vector store. If all paths are included in the
        cached paths, the hub does not need to be rebuilt.

        Args:
            paths (List[List[Triple]]): The current paths of the hub
                that represent the current content in the graph.
            hub_root_entity (EntityWithDirection): The root entity of the hub.

        Returns:
            Tuple[bool, List[HubPath]]: (needs_rebuild, cached_paths). When the hub
                does not need rebuilding the already-fetched cached paths are returned
                so the caller can reuse them without a second DB round-trip.
        """

        current_path_hashes = {path_to_hash(path) for path in paths}

        cached_hub_paths, _ = self.options.hub_storage_manager.get_all_hub_paths_from_hub(
            hub_entity_id=hub_root_entity.entity.uid
        )

        cached_path_hashes = {hub_path.path_hash for hub_path in cached_hub_paths}

        for path_hash in current_path_hashes:
            if path_hash not in cached_path_hashes:
                return True, []
        return False, cached_hub_paths

    def _build_hub_paths(self,
                         paths: List[List[Triple]],
                         hub_root_entity: EntityWithDirection) -> List[HubPath]:
        """
        Processes all paths of a hub in parallel and stores them in the vector store.

        The LLM-based text conversion for each path is parallelized via a thread pool.
        After conversion all texts are aggregated, deduplicated, embedded in a single
        batch call, and written to the vector store in a single batch upsert.

        Args:
            paths (List[List[Triple]]): List of paths where each path is a list of Triple objects.
            hub_root_entity (EntityWithDirection): The hub entity that is the root/center of
                these paths.

        Returns:
            List[HubPath]: The processed hub paths.
        """
        logger.debug("Processing %d paths..", len(paths))

        if not paths:
            return []

        # Before we build the hub paths we need to make sure that all previous
        # data is removed from the vector store. This is necessary to ensure
        # that no old data is left in the vector store.
        self.options.hub_storage_manager.delete_data_from_hub(
            hub_entity_id=hub_root_entity.entity.uid
        )

        progress_task = self.progress_handler.add_task(
            string_id=f"converting_hub_paths_{str(hub_root_entity.entity.uid)}",
            description=f"Converting Hub Paths for {hub_root_entity.entity.uid}",
            total=len(paths)
        )

        # Step 1: Convert all paths to text in parallel, using the shared LLM executor
        # when available (caps global OpenAI concurrency) or a local one as fallback.
        path_texts: List[str] = self.graph_converter.paths_to_text_batch(
            paths,
            max_concurrency=self.options.llm_workers,
            executor=self._llm_executor
        )
        self.progress_handler.update_task_by_string_id(
            progress_task, advance=len(paths))
        self.progress_handler.finish_by_string_id(progress_task)

        hub_paths: List[HubPath] = []
        for path, path_text in zip(paths, path_texts):
            if not path_text:
                raise ValueError(f"Failed to convert path to text. Path: {path}")
            hub_paths.append(HubPath(
                path_text=path_text,
                path_hash=path_to_hash(path),
                path=path
            ))

        # Vector storage manager (record builder, embeddings)
        self.options.hub_storage_manager.store_hub_batch(
            hub_root_entity=hub_root_entity,
            paths=paths,
            path_texts=path_texts
        )

        return hub_paths





