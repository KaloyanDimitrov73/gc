import os
from typing import List
import time
import json
from pydantic import BaseModel, Field, ConfigDict

from core.data.file_path_manager import FilePathManager
from core.data.models import Knowledge
from core.progress.progress_handler import ProgressHandler
from hublink.core.models.entity_with_direction import EntityWithDirection
from hublink.core.models.hub import IsHubOptions
from hublink.core.models.hub_link_settings import HubLinkSettings
from language_model.base.embedding_adapter import EmbeddingAdapter
from language_model.base.llm_adapter import LLMAdapter
from language_model import LLMStatTracker, LLMStats, LLMProvider
from knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from core.logging.logging import get_logger
from hublink.core.hub_storage_manager import HubStorageManager

from hublink.core.hub_finder import HubFinder
from hublink.indexing.hub_builder import HubBuilder, HubBuilderOptions
from language_model.config.embedding_config import EmbeddingConfig
from language_model.config.llm_config import LLMConfig

logger = get_logger(__name__)


class HubIndexerOptions(BaseModel):
    """
    Configuration options for the HubIndexer, controlling parallelism, model selection, and indexing parameters.

    Attributes:
        max_workers (int): Maximum number of workers for parallel processing.
        embedding_model (EmbeddingAdapter): Embedding model for vectorization.
        llm (LLMAdapter): LLM used for converting graph structures to text.
        hub_storage_manager (HubStorageManager): Manager that manage vector store for caching hub paths.
        max_indexing_depth (int): Maximum depth for indexing from root entities.
        is_hub_options (IsHubOptions): Options for hub classification.
        max_hub_path_length (int): Maximum allowed length for a hub path.
        distance_metric (str): Distance metric for vector store similarity.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    max_workers: int = Field(
        8,
        title="Max Workers",
        description="Maximum number of workers to use for parallel processing."
    )
    embedding_model: EmbeddingAdapter = Field(
        ...,
        title="Embedding Model",
        description="The model used to apply the embeddings."
    )
    llm: LLMAdapter = Field(
        ...,
        title="LLM Adapter",
        description="The LLM that is used for inference."
    )
    hub_storage_manager: HubStorageManager = Field(
        ...,
        title="Vector Store",
        description="The vector store for caching"
    )
    max_indexing_depth: int = Field(
        -1,
        title="Max Indexing Depth",
        description="The maximum depth to index."
    )
    is_hub_options: IsHubOptions = Field(
        ...,
        title="Is Hub Options",
        description="Options to determine if an entity is a hub."
    )
    max_hub_path_length: int = Field(
        ...,
        title="Max Hub Path Length",
        description="The maximum length of a hub path."
    )
    distance_metric: str = Field(
        "cosine",
        title="Distance Metric",
        description="The distance metric to use for the vector store."
    )

    @classmethod
    def from_settings(
            cls,
            settings: HubLinkSettings,
            hub_storage_manager: HubStorageManager,
            indexing_llm_config: LLMConfig,
            embedding_config:  EmbeddingConfig,
    ) -> "HubIndexerOptions":
        """
        Builds indexer options from a ``HubLinkSettings`` instance.

        Args:
            settings: Resolved HubLink settings.
            hub_storage_manager: The storage manager that manage indexing and retrieval.
            indexing_llm_config: The config for LLM used to build hub text for indexing.
            embedding_config: The config for embedding adapter to vectorize hub text.
        """
        _llm_provider = LLMProvider()

        return cls(
            embedding_model=_llm_provider.get_embeddings(embedding_config),
            llm=_llm_provider.get_llm_adapter(indexing_llm_config),
            hub_storage_manager=hub_storage_manager,
            max_workers=settings.max_workers,
            max_indexing_depth=settings.max_indexing_depth,
            is_hub_options=IsHubOptions(
                hub_edges=settings.hub_edges,
                types=settings.hub_types,
            ),
            max_hub_path_length=settings.max_hub_path_length,
            distance_metric=settings.distance_metric,
        )
class HubIndexer:
    """
    Indexes hubs in a knowledge graph for efficient retrieval.

    The HubIndexer traverses the knowledge graph from specified root entities, identifies hubs according to configuration,
    and stores their paths in a vector store for fast similarity search during retrieval.

    Args:
        graph (KnowledgeGraph): The knowledge graph to index.
        options (HubIndexerOptions): Configuration for indexing and model usage.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 options: HubIndexerOptions):
        self.graph = graph
        self.options = options
        self.progress_handler = ProgressHandler()
        self.hub_finder = HubFinder(
            graph=graph,
            is_hub_options=self.options.is_hub_options
        )
        self.hub_builder = HubBuilder(
            graph=graph,
            options=HubBuilderOptions(
                is_hub_options=self.options.is_hub_options,
                llm=self.options.llm,
                max_workers=self.options.max_workers,
                hub_storage_manager=self.options.hub_storage_manager,
                max_hub_path_length=self.options.max_hub_path_length
            )
        )

    def run_indexing(self,
                     root_entities: List[Knowledge],
                     force_index_update: bool = True):
        """
        Runs the indexing process on the graph. 
        The indexing process starts from a list of root entities. 
        It processes the graph by searching for all hubs that are on 
        paths from the root entities to the end of the graph. Each of 
        these hubs is then processed and stored in the vector store.

        Args:
            root_entities (List[EntitywithDirection]): The root entities
                to start the indexing from.
            force_index_update (bool): Whether to force update hubs that
                are already cached. If set to True, each hub needs to be 
                processed again allowing changes inside of hubs to be 
                reflected in the vector store.

        Further Notes:
            The reason we are indexing the hubs from root entities, is to 
            prevent having to process the whole graph. Graph can be very large
            with millions of entries. With this indexing approach, it is possible
            to index only a subgraph that is relevant for the user queries. 
            The subgraph can be carefully build using the hub classification 
            options and the maximal indexing depth.

        """
        # First we check whether the index has the distance metric that is defined
        # in the current config
        self.options.hub_storage_manager.ensure_distance_metric(distance_metric=self.options.distance_metric)
        '''
        # For tracking the stats of the LLM we prepare the
        # StatTracker
        llm_stat_tracker = LLMStatTracker()
        llm_stat_tracker.reset()

        # For tracking the emissions we prepare the EmissionTracker
        emission_tracker = EmissionTrackerManager()
        emission_tracker.start()
        '''
        # We start tracking the time here
        runtime = 0.0
        start_time = time.time()
        try:
            logger.info("Starting indexing...")
            logger.info(
                f"Embedding Model: {self.options.embedding_model.embedding_config.name_model}")
            logger.info(f"LLM: {self.options.llm.llm_config.name_model}")
            self._run_indexing_internal(
                root_entities=root_entities,
                force_index_update=force_index_update
            )
            logger.info("Indexing finished!")
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise e
        end_time = time.time()
        runtime = end_time - start_time
        self.options.hub_storage_manager.rebuild_index()

        '''
        # If no calls were made, no update to the index was made
        if llm_stat_tracker.stats.total_tokens == 0:
            self.options.hub_storage_manager.print_index_stats()
            return


        # After the indexing is done, we write the stats to a file
        self._write_indexing_stats(
            runtime=runtime,
            llm_stat_tracker=llm_stat_tracker,
            emission_tracker=emission_tracker
        )
        '''
    def _run_indexing_internal(self,
                               root_entities: List[Knowledge],
                               force_index_update: bool = True) -> None:
        """
        Internal method to run the indexing process.

        Args:
            root_entities (List[Knowledge]): The root entities to start the indexing from.
            force_index_update (bool): Whether to force update hubs that are already cached.
                If set to True, each hub needs to be processed again allowing changes
                inside of hubs to be reflected in the vector store.

        """
        max_indexing_depth = self.options.max_indexing_depth
        if max_indexing_depth == -1:
            max_indexing_depth = float('inf')

        # We initialize the roots to go in both directions of the graph
        root_entities_with_direction = [
            EntityWithDirection(entity=entity,
                                path_from_topic=[],
                                left=True)
            for entity in root_entities
        ]
        root_entities_with_direction.extend([
            EntityWithDirection(entity=entity,
                                path_from_topic=[],
                                left=False)
            for entity in root_entities
        ])

        logger.debug("Hub Root Entity: %s", root_entities_with_direction[0].entity.uid)
        logger.debug("Hub Root Entity text : %s", root_entities_with_direction[0].entity.text)
        logger.debug("Hub Root Entity type : %s", root_entities_with_direction[0].entity.knowledge_types)
        logger.debug("Hub Root Entity direction : %s", root_entities_with_direction[0].left)
        logger.debug("Hub Root Entity path : %s", root_entities_with_direction[0].path_from_topic)

        level = 0
        entities_to_start_traversal = root_entities_with_direction
        while level <= max_indexing_depth:

            # First we find all hubs starting from the root entities for the
            # current level.
            hub_entities, next_entities = self.hub_finder.find_hub_root_entities(
                root_entities=entities_to_start_traversal,
                reset_memory=False,
                need_same_hop_amount=False
            )
            entities_to_start_traversal = next_entities

            # Next we build the hubs for each hub entity
            next_entities, _ = self.hub_builder.build_hubs(
                hub_entities=hub_entities,
                update_cached_hubs=force_index_update
            )
            entities_to_start_traversal.extend(next_entities)

            if not entities_to_start_traversal:
                return
            level += 1

    def _write_indexing_stats(self,
                              runtime: float,
                              llm_stat_tracker: LLMStatTracker):
        """
        Writes statistics about the indexing process to a file.

        Args:
            runtime (float): The runtime of the indexing process.
            llm_stat_tracker (LLMStatTracker): The LLM stat tracker.
            #emission_tracker (EmissionTrackerManager): The emission tracker manager.
        """

        logger.debug("Writing indexing stats")
        vector_store_name = self.options.hub_storage_manager.vector_store_name()
        indexing_stats_file_path = FilePathManager().combine_paths(
            vector_store_name, "indexing_stats.json"
        )

        #emission_data: EmissionsTrackingData = emission_tracker.stop_and_get_results()
        llm_stats: LLMStats = llm_stat_tracker.get_stats()

        indexing_stats = {
            "runtime": runtime
        }
        indexing_stats.update(llm_stats.model_dump())
        #indexing_stats.update(emission_data.model_dump())
        embedding_dict = self.options.embedding_model.embedding_config.model_dump()
        embedding_dict = {f"embedding_model_{k}": v for k,
                          v in embedding_dict.items()}
        indexing_stats.update(embedding_dict)
        llm_dict = self.options.llm.llm_config.model_dump()
        llm_dict = {f"llm_{k}": v for k, v in llm_dict.items()}
        indexing_stats.update(llm_dict)

        # Load existing stats if available
        aggregated_stats = []
        if os.path.exists(indexing_stats_file_path):
            try:
                with open(indexing_stats_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        aggregated_stats = data
                    else:
                        aggregated_stats = [data]
            except Exception as e:
                logger.error(f"Failed to load existing stats: {e}")

        # Append the new stats record
        aggregated_stats.append(indexing_stats)

        with open(indexing_stats_file_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_stats, f, indent=4)
