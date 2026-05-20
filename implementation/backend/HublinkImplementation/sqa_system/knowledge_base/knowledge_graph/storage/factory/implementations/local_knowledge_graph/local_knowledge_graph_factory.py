from enum import Enum
from typing import Type, List
from typing_extensions import override


from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.data.models import PublicationDataset
from sqa_system.core.config.models.knowledge_base.knowledge_graph_config import KnowledgeGraphConfig
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.logging.logging import get_logger

from ....implementations.local_knowledge_graph import LocalKnowledgeGraph
from .local_graph_builder.rdflib_graph_builder import RDFLIBGraphBuilder
from .local_graph_builder.implementations.metadata_block import MetadataBlock
from .local_graph_builder.implementations.publisher_block import PublisherBlock
from .local_graph_builder.implementations.venue_block import VenueBlock
from .local_graph_builder.implementations.research_field_block import ResearchFieldBlock
from .local_graph_builder.implementations.authors_block import AuthorsBlock
from .local_graph_builder.implementations.additional_fields_block import AdditionalFieldsBlock
from .local_graph_builder.implementations.annotations_block import AnnotationsBlock
from .local_graph_builder.implementations.content_block import ContentBlock
from ...base.knowledge_graph_builder import KnowledgeGraphBuilder
logger = get_logger(__name__)


class BuildingBlock(Enum):
    """
    Enum class for the building blocks available for the creation of the graph.
    """
    METADATA = "metadata"
    AUTHORS = "authors"
    PUBLISHER = "publisher"
    VENUE = "venue"
    RESEARCH_FIELD = "research_field"
    ADDITIONAL_FIELDS = "additional_fields"
    ANNOTATIONS = "annotations"
    CONTENT = "content"


class LocalKnowledgeGraphFactory(KnowledgeGraphBuilder):
    """
    Factory class responsible to create a local rdflib knowledge graph
    """
    ADDITIONAL_CONFIG_PARAMS = [
        AdditionalConfigParameter(
            name="building_blocks",
            description="The building blocks to use for the graph creation",
            param_type=List[str],
            default_value=[block.value for block in BuildingBlock],
        )
    ]

    @classmethod
    @override
    def get_knowledge_graph_class(cls) -> Type[LocalKnowledgeGraph]:
        return LocalKnowledgeGraph

    @override
    def _create_knowledge_graph(self,
                                publications: PublicationDataset,
                                config: KnowledgeGraphConfig) -> LocalKnowledgeGraph:
        """
        Creates a local knowledge graph based on the specified configuration
        and dataset of publications.

        Args:
            publications: The dataset of publications to be used for creating or
                populating the knowledge graph
            config: The configuration for the knowledge graph

        Returns:
            LocalKnowledgeGraph: The created local knowledge graph object
        """
        logger.info("Creating RDF graph...")
        graph = LocalKnowledgeGraph(config)
        if self._try_to_load_serialized_graph(graph):
            return graph

        builder = self._prepare_builder(config)

        progress_handler = ProgressHandler()
        task_id = progress_handler.add_task("graph_creation",
                                            "Creating RDF graph...",
                                            total=len(publications.get_all_entries()))
        for publication in publications.get_all_entries():
            if publication.doi is None:
                logger.warning("Publication has no DOI. Skipping...")
                continue
            builder.build_publication_node(graph, publication)
            progress_handler.update_task_by_string_id(task_id, advance=1)
        progress_handler.finish_by_string_id(task_id)
        # serialize the graph
        self._serialize_graph(graph)
        return graph

    def _serialize_graph(self, graph: LocalKnowledgeGraph):
        """
        Saves the graph to a file to be used later.

        Args:
            graph: The graph to be serialized
        """
        file_path_manger = FilePathManager()
        path = graph.storage_path
        file_path_manger.ensure_dir_exists(path)
        graph.graph.serialize(destination=path, format='turtle')
        logger.info("RDF graph successfully created.")
        logger.debug(
            "The graph has been serialized to %s", path)

    def _try_to_load_serialized_graph(self,
                                      graph: LocalKnowledgeGraph) -> bool:
        """
        Tries to load a serialized graph from the specified path.

        Args:
            graph: The graph to load the serialized data into
        Returns:
            bool: True if the graph was successfully loaded, False otherwise
        """
        file_path_manger = FilePathManager()

        path_exists = file_path_manger.file_path_exists(graph.storage_path)
        if path_exists:
            graph.graph.parse(graph.storage_path, format='turtle')
            logger.info(
                f"RDF graph successfully loaded from file {graph.storage_path}.")
            return True
        return False

    def _get_extraction_llm_config(self, config: KnowledgeGraphConfig):
        """
        Returns the LLM config for the extraction of content from the publication.

        Args:
            config: The configuration for the knowledge graph

        Returns:
            LLMConfig: The LLM config for the extraction of content from the
                publication
        """
        extraction_llm_config = None
        if config.extraction_llm:
            extraction_llm_config = config.extraction_llm
            try:
                extraction_llm_config = LLMConfig.model_validate(
                    extraction_llm_config)
            except Exception as e:
                logger.error(f"Failed to parse extraction LLM config: {e}")
                extraction_llm_config = None
        else:
            logger.error(
                "No extraction LLM config found. Skipping extraction.")
        return extraction_llm_config

    def _prepare_builder(self, config: KnowledgeGraphConfig) -> RDFLIBGraphBuilder:
        """
        Generates a builder object for the graph and fills it with the
        specified building blocks which are specified in the config.

        Args:
            config: The configuration for the knowledge graph

        Returns:
            RDFLIBGraphBuilder: The builder object for the graph
        """
        builder = RDFLIBGraphBuilder()
        blocks = config.additional_params.get("building_blocks")

        if blocks is None:
            blocks = [block.value for block in BuildingBlock]

        if BuildingBlock.METADATA.value in blocks:
            builder.add_publication_block(MetadataBlock())
        if BuildingBlock.AUTHORS.value in blocks:
            builder.add_publication_block(AuthorsBlock())
        if BuildingBlock.PUBLISHER.value in blocks:
            builder.add_publication_block(PublisherBlock())
        if BuildingBlock.VENUE.value in blocks:
            builder.add_publication_block(VenueBlock())
        if BuildingBlock.RESEARCH_FIELD.value in blocks:
            builder.add_publication_block(ResearchFieldBlock())
        if BuildingBlock.ADDITIONAL_FIELDS.value in blocks:
            builder.add_publication_block(AdditionalFieldsBlock())
        if BuildingBlock.ANNOTATIONS.value in blocks:
            builder.add_publication_block(AnnotationsBlock())
        if BuildingBlock.CONTENT.value in blocks:
            builder.add_publication_block(ContentBlock(
                extraction_llm_config=self._get_extraction_llm_config(config)
            ))
        return builder
