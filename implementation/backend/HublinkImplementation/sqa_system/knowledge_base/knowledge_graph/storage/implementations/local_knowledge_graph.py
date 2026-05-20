import threading
import random
from typing import List, Set
from rdflib.namespace import RDF, RDFS, XSD
from rdflib import Graph, Namespace, URIRef
from typing_extensions import override
import pandas as pd

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.data.models.triple import Triple
from sqa_system.core.data.models.knowledge import Knowledge
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)

RE = Namespace("http://ressource.org/")
PRE = Namespace("http://predicate.org/")
SCHEMA = Namespace("http://schema.org/")
CLASS = Namespace("http://class.org/")
EX = Namespace("http://example.org/")

PREFIX_TO_NAMESPACE = {
    "re": RE,
    "pre": PRE,
    "schema": SCHEMA,
    "rdf": RDF,
    "rdfs": RDFS,
    "class": CLASS,
    "ex": EX,
    "xsd": XSD,
}


class LocalKnowledgeGraph(KnowledgeGraph):
    """
    This is a rdf graph implementation that uses the rdflib library
    to create a local knowledge graph.
    """

    ADDITIONAL_CONFIG_PARAMS = []

    @property
    def paper_type(self) -> str:
        return "http://schema.org/ScholarlyArticle"

    def __init__(self, config):
        super().__init__(config)
        self.graph = Graph()
        self.id_counters = {}
        self.lock = threading.Lock()
        self.prefixes = f"""
            PREFIX re: <{RE}>
            PREFIX schema: <{SCHEMA}>
            PREFIX rdf: <{RDF}>
            PREFIX rdfs: <{RDFS}>
            PREFIX class: <{CLASS}>
            PREFIX pre: <{PRE}>
            PREFIX ex: <{EX}>
            PREFIX xsd: <{XSD}>
        """
        self.entity_cache: List[str] = []
        fpm = FilePathManager()
        self.storage_path = fpm.combine_paths(fpm.KNOWLEDGE_GRAPH_DIR,
                                              "local_graph",
                                              config.config_hash,
                                              f"local_rdf_graph_{config.config_hash}.ttl")

    def build_entity_cache(self):
        """
        Builds a list of unique entities from the knowledge graph
        """
        query = """
        SELECT DISTINCT ?entity WHERE {
            ?entity a schema:ScholarlyArticle .
        }
        """
        result = self.run_sparql_query(query)
        self.entity_cache = result["entity"].tolist()

    @override
    def get_random_publication(self) -> Knowledge:
        """
        Returns a random entity from the knowledge graph using a cache.

        Returns:
            Knowledge: A random entity from the knowledge graph.
        """
        if len(self.entity_cache) == 0:
            self.build_entity_cache()
        if not self.entity_cache:
            raise ValueError("No entities found in the knowledge graph.")
        random_entity = random.choice(self.entity_cache)
        return self.get_entity_by_id(random_entity)

    def get_new_id(self, value: str) -> str:
        """
        Generates a new id for an entity.

        Args:
            value (str): The value to generate a new id for.

        Returns:
            str: The new id for the entity.
        """
        if value not in self.id_counters:
            self.id_counters[value] = 0
        else:
            self.id_counters[value] += 1
        return f"{value}_{self.id_counters[value]}"

    def run_sparql_query(self, query: str) -> pd.DataFrame:
        """
        Runs a SPARQL query on the knowledge graph and returns the results as a DataFrame.

        Args:
            query (str): The SPARQL query to run.

        Returns:
            pd.DataFrame: The results of the query as a DataFrame.
        """
        with self.lock:
            if not self.validate_sparql_query(query):
                return pd.DataFrame()
            try:
                query = self.prefixes + query
                results = self.graph.query(query)
            except Exception as e:
                logger.error(f"Error running SPARQL query: {str(e)}")
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

    @override
    def get_entities_by_predicate_id(self, predicates: Set[str]) -> Set[Knowledge]:
        entities = set()
        for pred in predicates:
            sparql_query = f"""
                SELECT DISTINCT ?entity
                WHERE {{
                    ?entity <{pred}> ?object .
                }}
            """
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
        query = f"""
            SELECT ?label WHERE {{
                <{entity_id}> rdfs:label ?label .
            }}
        """
        result = self.run_sparql_query(query)
        if not result.empty:
            name = result["label"].iloc[0]
        else:
            name = entity_id
        return Knowledge(uid=entity_id, text=name)

    @override
    def get_types_of_entity(self, entity: Knowledge) -> Set[str]:
        relations = self.get_relations_of_head_entity(entity)
        types = set()
        for relation in relations:
            if str(RDF.type) in relation.predicate:
                types.add(relation.entity_object.uid)
        return types

    @override
    def validate_graph_connection(self) -> bool:
        # Here we run a simple sparql query to check if the connection is valid
        query = "SELECT * WHERE { ?s ?p ?o . } LIMIT 1"
        try:
            self.run_sparql_query(query)
        # pylint: disable=broad-except
        except Exception:
            return False
        return True

    @override
    def get_main_triple_from_publication(self, publication_id: str) -> Triple | None:
        if not self.is_intermediate_id(publication_id):
            return None
        query = f"""
        SELECT ?label ?class WHERE {{
            OPTIONAL {{ <{publication_id}> rdfs:label ?label . }}
            OPTIONAL {{ <{publication_id}> a ?class . }}
        }}
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
        # If not class or label is found, check for any other triple
        query_other = f"""
        SELECT ?predicate ?object WHERE {{
            <{publication_id}> ?predicate ?object .
            FILTER(?predicate NOT IN (rdfs:label, rdf:type))
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
        query = f"""
        SELECT ?subject ?predicate ?subject_label ?predicate_label WHERE {{
            ?subject ?predicate <{knowledge.uid}> .
            OPTIONAL {{ ?subject rdfs:label ?subject_label . }}
            OPTIONAL {{ ?predicate rdfs:label ?predicate_label . }}
        }}
        """
        try:
            result = self.run_sparql_query(query)
        except Exception:
            return []
        relations = []
        for _, row in result.iterrows():
            text = row.get("subject_label", None)
            if text is None or text == "None":
                text = row.get("subject", None)
            head = Knowledge(
                text=text,
                uid=row["subject"]
            )
            description = row.get("predicate_label", None)
            if description is None or description == "None":
                description = row.get("predicate", None)
            if description is None:
                continue
            relation = Triple(
                predicate=description,
                entity_subject=head,
                entity_object=knowledge
            )
            relations.append(relation)
        return relations

    @override
    def get_relations_of_head_entity(self, knowledge: Knowledge) -> List[Triple]:
        query = f"""
        SELECT ?predicate ?object ?object_label ?predicate_label WHERE {{
            <{knowledge.uid}> ?predicate ?object .
            OPTIONAL {{ ?object rdfs:label ?object_label . }}
            OPTIONAL {{ ?predicate rdfs:label ?predicate_label . }}
        }}
        """
        try:
            result = self.run_sparql_query(query)
        except Exception:
            return []
        relations = []
        for _, row in result.iterrows():
            text = row.get("object_label", None)
            if text is None or text == "None":
                text = row.get("object", None)
            tail = Knowledge(
                text=text,
                uid=row["object"]
            )
            description = row.get("predicate_label", None)
            if description is None or description == "None":
                description = row.get("predicate", None)
            if description is None:
                continue
            relation = Triple(
                predicate=description,
                entity_subject=knowledge,
                entity_object=tail
            )
            relations.append(relation)
        return relations

    @override
    def is_publication_allowed_for_generation(self, publication: Knowledge) -> bool:
        return True

    def remove_prefix(self, uri: str) -> str:
        """
        Removes the namespace prefix from the given URI.
        """
        for _, namespace in PREFIX_TO_NAMESPACE.items():
            namespace_str = str(namespace)
            if uri.startswith(namespace_str):
                local_id = uri[len(namespace_str):]
                return local_id
        # We fallback to returning the whole URI if splitting fails
        return uri

    def get_namespace_short(self, uri: str) -> str:
        """
        Returns the namespace prefix for the given URI.

        Args:
            uri (str): The URI to get the namespace prefix for.

        Returns:
            str: The namespace prefix for the given URI.
        """
        for prefix, namespace in PREFIX_TO_NAMESPACE.items():
            namespace_str = str(namespace)
            if uri.startswith(namespace_str):
                return prefix
        return ""

    def get_full_uri(self, entity_id: str) -> URIRef:
        """
        Constructs the full URIRef for the given entity_id.
        If the entity_id is already a full URI, return it as is.

        Args:
            entity_id (str): The entity ID to construct the full URI for.

        Returns:
            URIRef: The full URIRef for the given entity_id.
        """
        if ":" in entity_id:
            prefix, local_part = entity_id.split(":", 1)
            local_part = local_part.replace(" ", "_")
            namespace = PREFIX_TO_NAMESPACE.get(prefix)
            if namespace:
                return namespace[self.get_new_id(local_part)]
            # Unknown prefix
            return URIRef(self.get_new_id(entity_id))
        # If no namespace matches, we assume its a local name and append to RE
        return PREFIX_TO_NAMESPACE["re"][self.get_new_id(entity_id)]

    @override
    def is_intermediate_id(self, entity_id: str) -> bool:
        if not entity_id:
            return False
        if str(RE) in entity_id:
            return True
        # a id is valid if it contans a : and then a _ followed by numbers
        is_valid_digit = False
        if ":" in entity_id:
            prefix, local_part = entity_id.split(":", 1)
            if prefix in PREFIX_TO_NAMESPACE:
                entity_id = str(PREFIX_TO_NAMESPACE[prefix][local_part])
            if "_" in entity_id:
                _, numbers = entity_id.split("_", 1)
                if numbers.isdigit():
                    is_valid_digit = True

        namespaces = set(PREFIX_TO_NAMESPACE.values())
        excluded = {RDF, RDFS, CLASS, SCHEMA, PRE}
        valid_namespaces = namespaces - excluded
        is_valid_namespace = any(entity_id.startswith(
            str(namespace)) for namespace in valid_namespaces)
        return is_valid_digit and is_valid_namespace
