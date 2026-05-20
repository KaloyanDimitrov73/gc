import re
import threading
from typing import List, Set
import random
from rdflib import Graph
from rdflib.term import _toPythonMapping
from rdflib.namespace import XSD
from typing_extensions import override
import yaml

import pandas as pd
from rdflib.namespace import RDF

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.data.models import Triple
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models.knowledge import Knowledge
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


def tolerant_integer_converter(lexical_value: str) -> int:
    """
    A converter to be able to convert float values that are stored as integers
    in the RDF graph. This is a workaround because we had errors thrown when loading
    the load the lokal ORKG graph as it seems that some values are stored as integers 
    but are actually float

    The workaround was found by chatGPT.

    Args:
        lexical_value (str): The lexical value to convert.

    Returns:
        int: The converted integer value.

    Raises:
        ValueError: If the conversion fails.    
    """
    try:
        return int(lexical_value)
    except ValueError:
        try:
            return int(float(lexical_value))
        except Exception as e:
            raise ValueError(
                f"Cannot convert '{lexical_value}' to integer") from e


class RDFFileGraph(KnowledgeGraph):
    """
    A class that that is initialized with a RDF file and loads the graph from it 
    using the rdflib library. The graph is then used to run SPARQL queries
    on the graph. 
    """

    @property
    def paper_type(self) -> str:
        return self.config.additional_params.get("paper_type", "Paper")

    def __init__(self, config):
        super().__init__(config)
        self._prepare_graph()
        self._prepare_queries()
        self.lock = threading.Lock()
        self.entity_cache = []
        self._prepare_prefixes()

    def _prepare_graph(self):
        # Register the custom converter for xsd:integer
        _toPythonMapping[XSD.integer] = tolerant_integer_converter
        self.graph = Graph()

    def _prepare_queries(self):
        """
        This method loads the SPARQL queries from YAML files which are used to implement
        the core functionalities that are required by the knowledge graph interface.

        Note: These queries can be changed in the future if using a different RDF graph.
        The current queries are based on the ORKG RDF graph.
        """
        fpm = FilePathManager()
        queries_folder = fpm.combine_paths(
            fpm.KNOWLEDGE_GRAPH_DIR,
            "rdf_file_graph",
            "sparql_queries"
        )
        self.get_entities_by_predicate_id_query = self._load_query_from_file(
            file_name="get_entities_by_predicate_id.yaml",
            queries_folder=queries_folder,
            fpm=fpm
        )
        self.get_relations_of_head_entity_query = self._load_query_from_file(
            file_name="get_relations_of_head_entity.yaml",
            queries_folder=queries_folder,
            fpm=fpm
        )
        self.get_relations_of_tail_entity_query = self._load_query_from_file(
            file_name="get_relations_of_tail_entity.yaml",
            queries_folder=queries_folder,
            fpm=fpm
        )
        self.get_entity_by_id_query = self._load_query_from_file(
            file_name="get_entity_by_id.yaml",
            queries_folder=queries_folder,
            fpm=fpm
        )

    def _load_query_from_file(self,
                              file_name: str,
                              queries_folder: str,
                              fpm: FilePathManager) -> str:
        """
        Helper function to load a SPARQL query from a YAML file.

        Args:
            file_name (str): The name of the YAML file containing the query.
            queries_folder (str): The folder where the query files are located.
            fpm (FilePathManager): The file path manager instance.
        Returns:
            str: The SPARQL query as a string.
        Raises:
            FileNotFoundError: If the query file is not found.
        """
        try:
            get_entities_by_predicate_path = fpm.combine_paths(
                queries_folder,
                file_name
            )
            with open(get_entities_by_predicate_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data["query"]
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_name}")
            raise FileNotFoundError(
                f"The query file {file_name} was not found."
                f"It should be located at {queries_folder}.") from e

    def _prepare_prefixes(self):
        """
        Prepares the prefixes used in SPARQL queries.

        This method loads prefixes from the configuration or uses default prefixes if none are provided.
        """
        loaded_prefixes = self.config.additional_params.get("prefixes", {})
        if not loaded_prefixes:
            loaded_prefixes = {
                "orkgr": "<http://orkg.org/orkg/resource/>",
                "orkgc": "<http://orkg.org/orkg/class/>",
                "orkgp": "<http://orkg.org/orkg/predicate/>",
                "rdfs": "<http://www.w3.org/2000/01/rdf-schema#>",
                "xsd": "<http://www.w3.org/2001/XMLSchema#>",
                "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
            }
            logger.info("No prefixes found in the graph configuration.")
            logger.info(f"Using default prefixes: {loaded_prefixes}")
        self.prefixes = loaded_prefixes
        self.prefixes_str = "\n".join(
            [f"PREFIX {prefix}: {namespace}" for prefix,
                namespace in self.prefixes.items()]
        )

    def run_sparql_query(self, query: str) -> pd.DataFrame:
        """
        Run a SPARQL query on the RDF graph and return the results as a pandas DataFrame.

        Args:
            query (str): The SPARQL query to run.

        Returns:
            pd.DataFrame: The results of the query as a pandas DataFrame.

        """
        with self.lock:
            if not self.validate_sparql_query(query):
                return pd.DataFrame()
            try:
                full_query = self.prefixes_str + "\n" + query
                results = self.graph.query(full_query)
            except Exception as e:
                print(f"Error running SPARQL query: {e}")
                return pd.DataFrame()
            data = []
            columns = results.vars
            for row in results:
                if isinstance(row, (list, tuple)):
                    data.append([str(cell) for cell in row])
                else:
                    data.append([str(row)])
            if isinstance(columns, list):
                df = pd.DataFrame(data, columns=[str(var) for var in columns])
            else:
                df = pd.DataFrame()
            return df

    def build_entity_cache(self):
        """
        Build a cache of entities using the configurable paper type.
        """
        query = f"""
        SELECT DISTINCT ?entity WHERE {{
            ?entity a <{self.paper_type}> .
        }}
        """
        result = self.run_sparql_query(query)
        if not result.empty and "entity" in result.columns:
            self.entity_cache = result["entity"].tolist()

    @override
    def get_random_publication(self) -> Knowledge:
        if not self.entity_cache:
            self.build_entity_cache()
        if not self.entity_cache:
            raise ValueError("No entities found in the knowledge graph.")
        random_entity = random.choice(self.entity_cache)
        return self.get_entity_by_id(random_entity)

    @override
    def get_entities_by_predicate_id(self, predicates: Set[str]) -> Set[Knowledge]:
        entities = set()
        for pred in predicates:
            sparql_query = self.get_entities_by_predicate_id_query.format(
                pred=pred
            )
            results_df = self.run_sparql_query(sparql_query)
            if results_df is not None:
                entities.update(self.get_entity_by_id(entity)
                                for entity in results_df["entity"])
        return entities

    @override
    def get_entity_ids_by_types(self, types: Set[str]) -> Set[str]:
        entities = set()
        for rdf_type in types:
            sparql_query = f"""
                SELECT DISTINCT ?entity
                WHERE {{
                    ?entity a <{rdf_type}> .
                }}
            """
            results_df = self.run_sparql_query(sparql_query)
            if results_df is not None:
                entities.update(str(entity) for entity in results_df["entity"])
        return entities

    @override
    def get_entity_by_id(self, entity_id: str) -> Knowledge | None:
        if not self.is_intermediate_id(entity_id):
            return entity_id
        query = self.get_entity_by_id_query.format(
            entity_id=entity_id
        )
        result = self.run_sparql_query(query)
        if not result.empty:
            name = result["tailEntity"].iloc[0]
        else:
            name = entity_id
        return Knowledge(uid=entity_id, text=name)

    @override
    def get_types_of_entity(self, entity: Knowledge) -> Set[str]:
        relations = self.get_relations_of_head_entity(entity)
        types = set()
        for relation in relations:
            if relation.predicate == RDF.type:
                types.add(relation.entity_object.uid)
        return types

    @override
    def get_main_triple_from_publication(self, publication_id: str) -> Triple | None:
        if not self.is_intermediate_id(publication_id):
            return None

        # First we try to get either the label or the class of the publication
        # If we find one of them, we return it as the main triple
        query = f"""
        SELECT ?label ?class WHERE {{
            OPTIONAL {{ <{publication_id}> rdfs:label ?label . }}
            OPTIONAL {{ <{publication_id}> a ?class . }}
        }} LIMIT 1
        """
        result = self.run_sparql_query(query)
        if not result.empty:
            if pd.notna(result["label"].iloc[0]):
                label = result["label"].iloc[0]
                return Triple(
                    entity_subject=Knowledge(
                        uid=publication_id, text=publication_id),
                    entity_object=Knowledge(uid=publication_id, text=label),
                    predicate="rdfs:label"
                )
            if pd.notna(result["class"].iloc[0]):
                cls = result["class"].iloc[0]
                return Triple(
                    entity_subject=Knowledge(
                        uid=publication_id, text=publication_id),
                    entity_object=Knowledge(uid=cls, text=cls),
                    predicate="rdf:type"
                )

        # If we don't find either, we try to get any other predicate and object
        # as the main triple
        query_other = f"""
        SELECT ?predicate ?object WHERE {{
            <{publication_id}> ?predicate ?object .
        }} LIMIT 1
        """
        result_other = self.run_sparql_query(query_other)
        if not result_other.empty:
            predicate = result_other["predicate"].iloc[0]
            object_ = result_other["object"].iloc[0]
            return Triple(
                entity_subject=Knowledge(
                    uid=publication_id, text=publication_id),
                entity_object=Knowledge(uid=object_, text=object_),
                predicate=predicate
            )
        return None

    @override
    def get_relations_of_tail_entity(self, knowledge: Knowledge) -> List[Triple]:
        query = self.get_relations_of_tail_entity_query.format(
            knowledge_id=knowledge.uid
        )
        result = self.run_sparql_query(query)
        relations: List[Triple] = []
        if result.empty:
            return relations
        for row in result.itertuples(index=False):
            relation = Triple(
                predicate=row.relation_description,
                entity_subject=Knowledge(
                    text=None if pd.isna(
                        row.subject_label) else str(row.subject_label),
                    uid=self._clean_id(str(row.subject)),
                ),
                entity_object=knowledge
            )
            relations.append(relation)
        return relations

    @override
    def get_relations_of_head_entity(self, knowledge: Knowledge) -> List[Triple]:
        query = self.get_relations_of_head_entity_query.format(
            knowledge_id=knowledge.uid
        )
        result = self.run_sparql_query(query)
        relations: List[Triple] = []
        if result.empty:
            return relations
        for row in result.itertuples(index=False):
            relation = Triple(
                predicate=row.relation_description,
                entity_subject=knowledge,
                entity_object=Knowledge(
                    text=None if pd.isna(
                        row.value_label) else str(row.value_label),
                    uid=self._clean_id(str(row.value)),
                )
            )
            relations.append(relation)

        return relations

    @override
    def is_publication_allowed_for_generation(self, publication: Knowledge) -> bool:
        return True

    @override
    def is_intermediate_id(self, entity_id: str) -> bool:
        if not entity_id:
            return False

        # If a custom regex is provided via the config, we use that
        id_regex = self.config.additional_params.get("is_id_regex", None)
        if id_regex:
            return re.fullmatch(id_regex, entity_id) is not None

        # Else we use the ORKG defaults
        if re.fullmatch(r"R\d+$", entity_id):
            return True
        return False

    @override
    def validate_graph_connection(self) -> bool:
        query = "SELECT * WHERE { ?s ?p ?o . } LIMIT 1"
        try:
            self.run_sparql_query(query)
        except Exception:
            return False
        return True

    def _clean_id(self, id_text: str) -> str:
        """
        Uses a regex to clean the text of an identifier for example by
        removing the prefix.

        Args:
            id_text (str): The identifier text to clean.

        Returns:
            str: The cleaned identifier text.
        """
        id_regex = self.config.additional_params.get("clean_id_regex", None)
        if id_regex:
            match = re.fullmatch(id_regex, id_text)
            if match:
                return match.group(1)

        # If no regex is provided, we use the default prefixes and remove those
        for prefix in self.prefixes.values():
            prefix = prefix.strip("<>")
            if id_text.startswith(prefix):
                return id_text[len(prefix):]

        return id_text
