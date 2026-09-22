from typing import Type
from typing_extensions import override
from sqa_system.core.config.models.knowledge_base.knowledge_graph_config import KnowledgeGraphConfig
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.logging.logging import get_logger

from ....implementations.rdf_file_graph import RDFFileGraph
from ...base.knowledge_graph_loader import KnowledgeGraphLoader
logger = get_logger(__name__)


class RDFFileGraphFactory(KnowledgeGraphLoader):
    """
    Factory class responsible for loading RDF file graphs.
    """
    ADDITIONAL_CONFIG_PARAMS = [
        AdditionalConfigParameter(
            name="graph_file_path",
            description=("The path to the RDF file to load."
                         "The file should be located in the directory "
                         "data/knowledge_base/knowledge_graphs/rdf_file_graph"),
            param_type=str,
            available_values=[]
        ),
        AdditionalConfigParameter(
            name="graph_file_format",
            description="The format of the RDF file to load",
            param_type=str,
            available_values=[]
        ),
        AdditionalConfigParameter(
            name="paper_type",
            description="The type in the graph that a root node of a paper has",
            param_type=str,
            default_value="Paper",
            available_values=[]
        ),
        AdditionalConfigParameter(
            name="prefixes",
            description="The prefixes to use in the graph",
            param_type=dict,
            default_value={
                "orkgr": "<http://orkg.org/orkg/resource/>",
                "orkgc": "<http://orkg.org/orkg/class/>",
                "orkgp": "<http://orkg.org/orkg/predicate/>",
                "rdfs": "<http://www.w3.org/2000/01/rdf-schema#>",
                "xsd": "<http://www.w3.org/2001/XMLSchema#>",
                "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
            },
            available_values=[]
        ),
        AdditionalConfigParameter(
            name="is_id_regex",
            description=(
                "A regex that checks an id of an entity in the graph and should return true "
                "if it is a valid id in the graph."
            ),
            param_type=str,
            available_values=[]
        ),
        AdditionalConfigParameter(
            name="clean_id_regex",
            description=(
                "A regex that processes IDs retrieved from the graph to remove "
                "unnecessary parts. If not provided, the defined prefixes are "
                "removed by default."
            ),
            param_type=str,
            available_values=[]
        )
    ]

    @classmethod
    @override
    def get_knowledge_graph_class(cls) -> Type[RDFFileGraph]:
        return RDFFileGraph

    @override
    def _load_knowledge_graph(self, config: KnowledgeGraphConfig) -> RDFFileGraph:
        """
        Main class to load a existing RDF file graph.
        It loads the graph from the file path provided in the configuration.

        Args:
            config (KnowledgeGraphConfig): The configuration for the knowledge graph.

        Returns:
            RDFFileGraph: The loaded RDF file graph.
        """
        graph_path = config.additional_params.get("graph_file_path", None)
        if graph_path is None:
            raise ValueError(
                "Graph file path is missing in the configuration.")

        graph_format = config.additional_params.get("graph_file_format", None)
        if graph_format is None:
            raise ValueError(
                "Graph file format is missing in the configuration.")

        fpm = FilePathManager()
        final_graph_path = fpm.combine_paths(
            fpm.KNOWLEDGE_GRAPH_DIR,
            "rdf_file_graph",
            graph_path)
        if not fpm.file_path_exists(final_graph_path):
            raise FileNotFoundError(
                f"Graph file {final_graph_path} not found.")
        try:
            graph = RDFFileGraph(config=config)
            graph.graph.parse(final_graph_path, format=graph_format)
            return graph
        except Exception as e:
            logger.error("There was an error loading or parsing the RDF file.")
            raise ValueError(f"Error loading graph file: {e}") from e
