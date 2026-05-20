import re
from typing import List, Set, Tuple
import random
import json
import time
import hashlib
import requests
from typing_extensions import override

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.data.cache_manager import CacheManager
from sqa_system.knowledge_base.knowledge_graph.storage.utils import GraphPathFilter
from sqa_system.core.data.models import Triple, Subgraph
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models.knowledge import Knowledge
from sqa_system.core.config.models.knowledge_base.knowledge_graph_config import KnowledgeGraphConfig
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


class ORKGRemoteGraph(KnowledgeGraph):
    """
    A remote knowledge graph implementation that uses the ORKG. It is implemented using the ORKG REST API.
    The documentation that this implementation is based on is here: https://tibhannover.gitlab.io/orkg/orkg-backend/api-doc/
    The ORKG is a scholarly knowledge graph that is accessible here https://sandbox.orkg.org/. Note that we are
    using the sandbox environment by default, but this can be changed in the config.

    Args:
        config (KnowledgeGraphConfig): The configuration for the knowledge graph.
    """
    TIMEOUT = 15

    @property
    def root_type(self) -> str:
        return "entity_type"

    @property
    def paper_type(self) -> str:
        """
        The root type identifier for this knowledge graph.
        """
        return "Paper"

    def __init__(self, config: KnowledgeGraphConfig):
        super().__init__(config)
        self.publications_root_cache: List[Knowledge] = []
        self.cache_manager = CacheManager()
        self.only_subgraph_mode = True
        self._prepare_values(config)

    def _prepare_values(self, config: KnowledgeGraphConfig):
        """
        Prepares values for the ORKG remote graph.

        Args:
            config (KnowledgeGraphConfig): The configuration for the knowledge graph.
        """
        self.cache_subgraph_key = "orkg_subgraph"
        self.cache_publication_roots_key = "orkg_publications_root_cache"
        if config:
            self.cache_subgraph_key += "_" + config.config_hash
            self.cache_publication_roots_key += "_" + config.config_hash
        if config is not None:
            self.base_url = config.additional_params.get(
                "orkg_base_url", "https://sandbox.orkg.org")
            contributions = config.additional_params.get(
                "contribution_building_blocks", {"Publication Overview": None})
            self.contribution_names = contributions.keys()
            self.subgraph_root_entity_id = config.additional_params.get(
                "subgraph_root_entity_id", "R659055")
            self.publication_limit = config.additional_params.get(
                "limit_publications", -1)
        else:
            self.base_url = "https://sandbox.orkg.org"
            self.contribution_names = "Publication Overview"
            self.subgraph_root_entity_id = "R659055"
            self.publication_limit = -1

    def _make_get_request(self,
                          url: str,
                          params: dict = None,
                          headers: dict = None,
                          retry_amount: int = 20) -> requests.Response:
        """
        Helper method that performs a GET request with a retry mechanism.

        Args:
            url (str): The URL to send the GET request to.
            params (dict): The parameters to include in the GET request.
            headers (dict): The headers to include in the GET request.
            retry_amount (int): The number of times to retry the request if it fails.

        Returns:
            requests.Response: The response from the GET request.
        """
        attempt = 0
        if headers is None:
            headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "Accept": "application/json"
            }
        while attempt < retry_amount:
            try:
                response = requests.get(
                    url, params=params, headers=headers, timeout=self.TIMEOUT)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                attempt += 1
                backoff_value = min(2 ** attempt, 120)
                jitter_value = backoff_value * 0.4
                sleep_time = backoff_value + random.uniform(-jitter_value, jitter_value)
                logger.warning(
                    f"GET request failed for {url} with params {params}. Attempt {attempt} of "
                    f"{retry_amount} waiting {round(sleep_time/60, 2)} mins. Error: {e}")
                time.sleep(backoff_value + random.uniform(-jitter_value, jitter_value))

                if attempt >= retry_amount:
                    raise e
        return None

    def update_cache_if_not_exists(self):
        """
        Looks up the cached data and only updates the cache if the data is not
        already cached.
        """
        if self.config is None:
            return
        if self.cache_manager.get_table(self.cache_subgraph_key):
            logger.debug("Cache already exists, skipping update.")
            return

        # If the cache does not exist, we need to create it
        logger.info("ORKG Cache does not exist, creating it.")
        self.cache_subgraph()

    def cache_subgraph(self,
                       blacklist_predicates: List[str] = None):
        """
        Runs a caching process where a subgraph is built based on the given root
        entity. Each triple in the subgraph is then cached in a database.

        Every other function of the ORKGRemoteGraph will only return triples that
        are contained in the cache to make sure, that only specific parts of the
        ORKG graph are considered.

        Args:
            blacklist_predicates (List[str]): A list of predicates that should not be
                further traversed.
        """
        logger.info("Caching ORKG subgraph")
        self.only_subgraph_mode = False
        # First we delete previous data
        self.cache_manager.delete_table(self.cache_subgraph_key)
        self.cache_manager.delete_table(self.cache_publication_roots_key)
        root_entity_id = self.subgraph_root_entity_id

        if blacklist_predicates is None:
            # To not go to another research field
            blacklist_predicates = ["has subfield"]

        root_entity = self.get_entity_by_id(root_entity_id)
        if not root_entity:
            raise ValueError(
                f"Root entity with id {root_entity_id} not found.")

        subgraphs: List[Tuple[Knowledge, Subgraph]] = self._get_neighbor_subgraphs(
            root_entity=root_entity,
            blacklist_predicates=blacklist_predicates
        )

        ProgressHandler().add_task(
            string_id="orkg_subgraph_caching",
            description="ORKG Subgraph Caching..",
            total=len(subgraphs),
            reset=True
        )
        for neighbor_entity, subgraph in subgraphs:
            if (self.publications_root_cache and
                self.publication_limit > 0 and
                    self.publication_limit <= len(self.publications_root_cache)):
                break
            self._process_subgraph_for_caching(subgraph, neighbor_entity)
            ProgressHandler().update_task_by_string_id("orkg_subgraph_caching", 1)

        ProgressHandler().finish_by_string_id("orkg_subgraph_caching")
        logger.info("Finished caching ORKG subgraph")
        self._save_cached_subgraph_to_file()
        self.only_subgraph_mode = True

    def _get_neighbor_subgraphs(
        self,
        root_entity: Knowledge,
        blacklist_predicates: List[str] = None) -> List[Tuple[Knowledge, Subgraph]]:
        """
        For the given root all neighbour entities are retrieved and their subgraphs
        are fetched. The subgraphs are then returned as a list of tuples, where each
        tuple contains the neighbour entity and its subgraph.

        Args:
            root_entity (Knowledge): The root entity to start from. It gets all the
                neighbours and then builds the subgraph with the neighbour entity
                as the root.
            blacklist_predicates (List[str]): A list of predicates that should not be
                further traversed.

        Returns:
            List[Tuple[Knowledge, Subgraph]]: A list of tuples, where each tuple
                contains the neighbour entity and its subgraph.
        """
        neighbors = self.get_relations_of_tail_entity(root_entity)
        subgraphs: List[Tuple[Knowledge, Subgraph]] = []

        for triple in neighbors:

            if any(triple.predicate.lower() in predicate.lower() for predicate in blacklist_predicates):
                continue

            if triple.entity_subject.uid == root_entity.uid:
                neighbor_entity = triple.entity_object
            else:
                neighbor_entity = triple.entity_subject

            neighbor_subgraph = self._get_subgraph(neighbor_entity.uid)
            subgraphs.append((neighbor_entity, neighbor_subgraph))
        return subgraphs

    def _process_subgraph_for_caching(self, subgraph: Subgraph, root_entity: Knowledge):
        """
        This function caches the triples of a subgraph. It is essential to implement a filtering
        mechanism to only allow retrievers to access the triples that are relevant for
        our experiments. This is required because the ORKG graph is very large and
        contains a lot of data that is not relevant for our experiments.

        Args:
            subgraph (Subgraph): The subgraph to process.
            root_entity (Knowledge): The root entity of the subgraph.
        """
        publication_candidate_entity: Knowledge = None
        includes_contribution = False
        triples_to_cache: List[Triple] = []
        entities_to_cache: List[Knowledge] = []

        # First we filter out all contributions from the paper that are
        # not in the list of contributions we are interested in
        filtered_by_contri = GraphPathFilter().filter_paths_by_type_and_name(
            filter_type="Contribution",
            keep_names=self.contribution_names,
            root=root_entity,
            subgraph=subgraph,
        )

        # Sometimes old deleted data is retrieved, we need to remove this as well
        filtered_by_deleted = GraphPathFilter().filter_paths_by_type_substring(
            root=root_entity,
            subgraph=filtered_by_contri,
            filter_type="deleted"
        )

        # Now we are going to collect all triples and entities
        # and check whether the data is a publication and if it
        # contains the required contribution
        for triple in filtered_by_deleted:

            triples_to_cache.append(triple)
            entities_to_cache.append(triple.entity_subject)
            entities_to_cache.append(triple.entity_object)

            # Get the publication root
            if (self.paper_type in triple.entity_object.knowledge_types or
                    self.paper_type in triple.entity_subject.knowledge_types):
                publication_candidate_entity = triple.entity_subject

            # Check for the contribution with the name of our contrib
            if (triple.entity_object.text in self.contribution_names
                    or triple.entity_subject.text in self.contribution_names):
                includes_contribution = True

        # We are going to cache the data only if its not a publication or if
        # it is a publication and contains the contribution
        if not publication_candidate_entity or (publication_candidate_entity and includes_contribution):
            for triple in triples_to_cache:
                self.cache_manager.add_data(
                    meta_key=self.cache_subgraph_key,
                    dict_key=self._get_hash_value(triple.model_dump_json()),
                    value=triple.model_dump_json(),
                    silent=True
                )
            for entity in entities_to_cache:
                self.cache_manager.add_data(
                    meta_key=self.cache_subgraph_key,
                    dict_key=self._get_hash_value(
                        entity.model_dump_json()),
                    value=entity.model_dump_json(),
                    silent=True
                )
            # Populate a seperate chache for the publication roots
            if publication_candidate_entity:
                self.cache_manager.add_data(
                    meta_key=self.cache_publication_roots_key,
                    dict_key=self._get_hash_value(
                        publication_candidate_entity.model_dump_json()),
                    value=publication_candidate_entity.model_dump_json(),
                    silent=True
                )
                self.publications_root_cache.append(
                    publication_candidate_entity)

    def _save_cached_subgraph_to_file(self):
        """
        Saves the cached subgraph to a JSON file and writes a companion
        metadata file (<hash>.meta.json) recording when the cache was built
        and how many entries it contains.
        """
        from datetime import datetime, timezone
        fpm = FilePathManager()
        file_path = fpm.combine_paths(fpm.KNOWLEDGE_GRAPH_DIR,
                                      "orkg",
                                      self.config.config_hash + ".json")
        meta_path = fpm.combine_paths(fpm.KNOWLEDGE_GRAPH_DIR,
                                      "orkg",
                                      self.config.config_hash + ".meta.json")
        fpm.ensure_dir_exists(file_path)

        cached_data = self.cache_manager.get_table(
            meta_key=self.cache_subgraph_key)
        if not cached_data:
            logger.debug("Found no cached data to save the ORKG graph to.")

        json_objects = []
        for _, value in cached_data.items():
            json_data = json.loads(value)
            json_objects.append(json_data)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(json_objects, file, indent=4)

        meta = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(json_objects),
            "orkg_base_url": self.base_url,
            "subgraph_root_entity_id": self.subgraph_root_entity_id,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

        logger.info(f"Saved cached subgraph to {file_path} ({len(json_objects)} entries)")

    @override
    def get_random_publication(self) -> Knowledge:
        self._init_publication_cache()
        random_entity = random.choice(self.publications_root_cache)
        return random_entity

    @override
    def is_publication_allowed_for_generation(self, publication: Knowledge) -> bool:
        self._init_publication_cache()
        return publication in self.publications_root_cache

    def _init_publication_cache(self):
        if not self.publications_root_cache:
            all_data = self.cache_manager.get_table(
                self.cache_publication_roots_key)
            if not all_data:
                raise ValueError("No publication roots cached.")
            for _, json_data in all_data.items():
                self.publications_root_cache.append(
                    Knowledge.model_validate_json(json_data))

    def _get_cached_triples_and_entities(self) -> Tuple[List[Triple], List[Knowledge]]:
        """
        Deserializes all entries from the sqlite subgraph cache into Triple and Knowledge objects.

        This helper is used by traversal methods when ``only_subgraph_mode=True`` so that
        graph traversal can be served entirely from the local sqlite cache without making any
        HTTP requests to the ORKG REST API.

        The cache stores serialized JSON objects. An entry is a Triple if it contains a
        ``"predicate"`` key; otherwise it is treated as a Knowledge node.

        Returns:
            Tuple[List[Triple], List[Knowledge]]: All cached triples and entity nodes.
        """
        all_data = self.cache_manager.get_table(self.cache_subgraph_key) or {}
        triples: List[Triple] = []
        entities: List[Knowledge] = []
        for _, value in all_data.items():
            try:
                obj = json.loads(value)
                if "predicate" in obj:
                    triples.append(Triple.model_validate(obj))
                else:
                    entities.append(Knowledge.model_validate(obj))
            except Exception:
                continue
        return triples, entities

    @override
    def validate_graph_connection(self) -> bool:
        """
        Checks if the ORKG REST API is reachable by trying to fetch a single statement.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/statements",
                params={"page": 0, "size": 1},
                headers={
                    "Content-Type": "application/json;charset=UTF-8",
                    "Accept": "application/json"
                },
                timeout=self.TIMEOUT
            )
            response.raise_for_status()
            return True
        except Exception as e:
            raise ValueError(
                f"Failed to connect to the ORKG REST API: {e}") from e

    @override
    def get_main_triple_from_publication(self, publication_id: str) -> Triple | None:
        if not self.is_intermediate_id(publication_id):
            return None
        relations = self.get_relations_of_head_entity(
            self.get_entity_by_id(publication_id))

        if not relations:
            return None

        for relation in relations:
            if relation.predicate == "doi":
                return relation
        return None

    @override
    def get_entity_by_id(self, entity_id: str) -> Knowledge | None:
        """
        Retrieves a Knowledge entity by its ORKG ID.

        When ``only_subgraph_mode=True`` (the default during normal operation), the entity is
        looked up directly in the local sqlite subgraph cache, avoiding any HTTP request to the
        ORKG REST API.  This is safe because the cache was built from the same ORKG data during
        ``cache_subgraph()``.

        When ``only_subgraph_mode=False`` (set only inside ``cache_subgraph()`` while the cache
        is being populated), the live ORKG REST API is queried as before.

        Args:
            entity_id (str): The ORKG resource ID (e.g. ``"R659055"``).

        Returns:
            Knowledge | None: The corresponding Knowledge node, or ``None`` if not found.
        """
        if not self.is_intermediate_id(entity_id):
            return None
        if self.only_subgraph_mode and self.config is not None:
            triples, entities = self._get_cached_triples_and_entities()
            for e in entities:
                if e.uid == entity_id:
                    return e
            # Fallback: the JSON cache only stores triples (no standalone Knowledge nodes),
            # so search the subject/object of cached triples as well.
            for triple in triples:
                if triple.entity_subject.uid == entity_id:
                    return triple.entity_subject
                if triple.entity_object.uid == entity_id:
                    return triple.entity_object
            return None
        statement = self._rest_get_resource_by_id(entity_id)
        if not statement:
            return None
        return Knowledge(
            text=statement.get("label", entity_id),
            uid=entity_id,
            knowledge_types=statement.get("classes", []))

    @override
    def get_relations_of_head_entity(self, knowledge: Knowledge) -> List[Triple]:
        """
        Returns all triples where the given entity is the subject (head).

        When ``only_subgraph_mode=True`` (the default during normal operation), triples are
        filtered directly from the local sqlite subgraph cache, avoiding any HTTP request to the
        ORKG REST API.

        When ``only_subgraph_mode=False`` (set only inside ``cache_subgraph()``), the live ORKG
        REST API is queried and results are filtered through ``_check_if_triple_allowed()``.

        Args:
            knowledge (Knowledge): The head entity whose outgoing relations are requested.

        Returns:
            List[Triple]: All cached triples with ``entity_subject.uid == knowledge.uid``.
        """
        if self.only_subgraph_mode and self.config is not None:
            triples, _ = self._get_cached_triples_and_entities()
            return [t for t in triples if t.entity_subject.uid == knowledge.uid]
        params = {"subject_id": knowledge.uid, "page": 0, "size": 100}
        statements = self._get_all(params)
        relations: List[Triple] = []
        for stmt in statements:
            predicate_label = stmt.get("predicate", {}).get("label", "")
            object_data = stmt.get("object", {})
            obj_uid = object_data.get("id", "")
            obj_label = object_data.get("label")
            obj_classes = object_data.get("classes", [])
            triple = Triple(
                predicate=predicate_label,
                entity_subject=knowledge,
                entity_object=Knowledge(
                    text=obj_label, uid=obj_uid, knowledge_types=obj_classes)
            )
            if not self._check_if_triple_allowed(triple):
                continue
            relations.append(triple)
        return relations

    @override
    def get_relations_of_tail_entity(self, knowledge: Knowledge) -> List[Triple]:
        """
        Returns all triples where the given entity is the object (tail).

        When ``only_subgraph_mode=True`` (the default during normal operation), triples are
        filtered directly from the local sqlite subgraph cache, avoiding any HTTP request to the
        ORKG REST API.

        When ``only_subgraph_mode=False`` (set only inside ``cache_subgraph()``), the live ORKG
        REST API is queried and results are filtered through ``_check_if_triple_allowed()``.

        Args:
            knowledge (Knowledge): The tail entity whose incoming relations are requested.

        Returns:
            List[Triple]: All cached triples with ``entity_object.uid == knowledge.uid``.
        """
        if self.only_subgraph_mode and self.config is not None:
            triples, _ = self._get_cached_triples_and_entities()
            return [t for t in triples if t.entity_object.uid == knowledge.uid]
        params = {"object_id": knowledge.uid, "page": 0, "size": 100}
        statements = self._get_all(params)
        relations: List[Triple] = []
        for stmt in statements:
            predicate_label = stmt.get("predicate", {}).get("label", "")
            subject_data = stmt.get("subject", {})
            subj_uid = subject_data.get("id", "")
            subj_label = subject_data.get("label")
            subj_classes = subject_data.get("classes", [])
            triple = Triple(
                predicate=predicate_label,
                entity_subject=Knowledge(
                    text=subj_label, uid=subj_uid, knowledge_types=subj_classes),
                entity_object=knowledge
            )
            if not self._check_if_triple_allowed(triple):
                continue
            relations.append(triple)
        return relations

    @override
    def get_types_of_entity(self, entity: Knowledge) -> Set[str]:
        return entity.knowledge_types

    def _rest_get_resource_by_id(self, resource_id: str) -> dict:
        """
        Fetches a single resource by its ID using the REST API.

        Args:
            resource_id (str): The ID of the resource to fetch.
        Returns:
            dict: The resource data as a dictionary.
        """
        try:
            response = self._make_get_request(
                f"{self.base_url}/api/resources/{resource_id}")
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching resource {resource_id}: {e}")
            return {}

    def _get_subgraph(self, root_id: str) -> Subgraph:
        """
        Starting from a root entity, fetches all statements that are connected to it
        in the direction of the graph.

        Args:
            root_id (str): The ID of the root entity.

        Returns:
            Subgraph: A subgraph containing all statements connected to the root entity.
        """
        try:
            response = self._make_get_request(
                f"{self.base_url}/api/statements/{root_id}/bundle")
            response_as_json = response.json()
            return self._convert_statements_to_subgraph(response_as_json)
        except Exception as e:
            logger.error(f"Error fetching resource {root_id}: {e}")
            return {}

    def _convert_statements_to_subgraph(self, data: dict) -> Subgraph:
        """
        Converts the return from the ORKG API to a subgraph object that the SQA system can use.

        Args:
            data (dict): The data returned from the ORKG API.

        Returns:
            Subgraph: A subgraph object containing the triples.
        """
        try:
            triples: List[Triple] = []
            for statement in data.get("statements", []):
                subject_json = statement.get("subject", {})
                subject_entity = Knowledge(
                    uid=subject_json.get("id"),
                    text=subject_json.get("label"),
                    knowledge_types=subject_json.get("classes", [])
                )
                object_json = statement.get("object", {})
                object_entity = Knowledge(
                    uid=object_json.get("id"),
                    text=object_json.get("label"),
                    knowledge_types=object_json.get("classes", [])
                )
                predicate_json = statement.get("predicate", {})
                triple = Triple(
                    entity_subject=subject_entity,
                    entity_object=object_entity,
                    predicate=predicate_json.get("label")
                )
                triples.append(triple)
            return Subgraph(root=triples)
        except Exception as e:
            logger.error(f"Error converting statements to subgraph: {e}")
            raise e

    def _get_all(self, params: dict, should_be_statements: bool = True) -> List[dict]:
        """
        Fetches all statements matching the given parameters.

        Args:
            params (dict): The parameters to filter the statements.
            should_be_statements (bool): If True, fetches statements. If False, fetches resources.
        Returns:
            List[dict]: A list of statements or resources.
        """
        all_statements = []
        page = params.get("page", 0)
        page_size = params.get("size", 100)
        while True:
            params.update({"page": page, "size": page_size})
            try:
                if should_be_statements:
                    response_url = f"{self.base_url}/api/statements"
                else:
                    response_url = f"{self.base_url}/api/resources"
                response = self._make_get_request(response_url, params=params)
                data = response.json()
                statements = data.get("content", [])
                page_info = data.get("page", {})
                total_pages = page_info.get("total_pages", 1)
            except Exception as e:
                logger.error(
                    f"Error fetching statements with params {params}: {e}")
                break

            if not statements:
                break

            all_statements.extend(statements)
            if page >= total_pages - 1:
                break
            page += 1

        return all_statements

    @override
    def get_entities_by_predicate_id(self, predicates: Set[str]) -> Set[Knowledge]:
        """
        Retrieves entities that have the specified predicates.
        For each predicate in the set, all statements with that predicate are fetched.
        The subject ID of each matching statement is returned.

        Args:
            predicates (Set[str]): A set of predicate IDs.
        Returns:
            Set[Knowledge]: A set of Knowledge objects representing the entities.
        """
        entities = set()
        for pred in predicates:
            params = {"predicate_id": pred, "page": 0, "size": 100}
            statements = self._get_all(params)
            for stmt in statements:
                subject_data = stmt.get("subject", {})
                subj_id = subject_data.get("id")
                subj_label = subject_data.get("label")
                subj_classes = subject_data.get("classes", [])
                subj_knowledge = Knowledge(
                    text=subj_label, uid=subj_id, knowledge_types=subj_classes)
                if self._check_if_knowledge_allowed(subj_knowledge) and subj_id:
                    entities.add(subj_knowledge)
        return entities

    @override
    def get_entity_ids_by_types(self, types: Set[str]) -> Set[str]:
        """
        Retrieves IDs of all entities whose ``knowledge_types`` intersect with the given types.

        When ``only_subgraph_mode=True`` (the default during normal operation), entities are
        read directly from the local sqlite subgraph cache, avoiding any HTTP request to the
        ORKG REST API.  Only Knowledge objects (not Triples) are considered, and each must pass
        ``_check_if_knowledge_allowed()``.

        When ``only_subgraph_mode=False`` (set only inside ``cache_subgraph()``), the live ORKG
        REST API is queried via the ``/api/resources`` endpoint with an ``include`` filter.

        Args:
            types (Set[str]): A set of ORKG class/type identifiers (e.g. ``{"Paper"}``).

        Returns:
            Set[str]: IDs of all matching entities present in the subgraph cache.
        """
        if self.only_subgraph_mode and self.config is not None:
            _, entities = self._get_cached_triples_and_entities()
            result: Set[str] = set()
            for e in entities:
                if e.uid and types.intersection(e.knowledge_types or set()):
                    if self._check_if_knowledge_allowed(e):
                        result.add(e.uid)
            return result
        entities_set: Set[str] = set()
        params = {
            "include": ",".join(types),
            "page": 0,
            "size": 100
        }
        resources = self._get_all(params, should_be_statements=False)
        for resource in resources:
            subj_id = resource.get("id")
            subj_label = resource.get("label")
            subj_classes = resource.get("classes", [])
            subj_knowledge = Knowledge(
                text=subj_label, uid=subj_id, knowledge_types=subj_classes)
            if self._check_if_knowledge_allowed(subj_knowledge) and subj_id:
                entities_set.add(subj_id)
        return entities_set

    @override
    def is_intermediate_id(self, entity_id: str) -> bool:
        """
        Checks if an entity ID is an ORKG ID.

        Args:
            entity_id (str): The entity ID.

        Returns:
            bool: True if the entity ID is an ORKG ID, False otherwise.
        """
        if entity_id is None:
            return False
        contains_prefix = "orkgr:" in entity_id or "orkgp:" in entity_id or "orkgc:" in entity_id
        id_pattern = re.compile(r"([R]\d+)")
        has_regex_match = bool(id_pattern.search(entity_id))
        return contains_prefix or has_regex_match

    def _check_if_triple_allowed(self, triple: Triple) -> bool:
        """
        Checks if a triple is part of the cached subgraph.

        Args:
            triple (Triple): The triple to check.
        Returns:
            bool: True if the triple is part of the cached subgraph, False otherwise.
        """
        if self.config is None or not self.only_subgraph_mode:
            return True
        hash_value = self._get_hash_value(triple.model_dump_json())
        cached_triple = self.cache_manager.get_data(
            meta_key=self.cache_subgraph_key,
            dict_key=hash_value,
            silent=True
        )

        return cached_triple is not None

    def _check_if_knowledge_allowed(self, knowledge: Knowledge) -> bool:
        """
        Checks if a knowledge object is part of the cached subgraph.

        Args:
            knowledge (Knowledge): The knowledge object to check
        Returns:
            bool: True if the knowledge object is part of the cached subgraph, False otherwise.
        """
        if self.config is None or not self.only_subgraph_mode:
            return True
        hash_value = self._get_hash_value(knowledge.model_dump_json())
        cached_knowledge = self.cache_manager.get_data(
            meta_key=self.cache_subgraph_key,
            dict_key=hash_value,
            silent=True
        )

        return cached_knowledge is not None

    def _get_hash_value(self, text: str) -> str:
        """
        Converts a string to a hash value using MD5.
        Args:
            text (str): The string to hash.
        Returns:
            str: The MD5 hash of the string.
        """
        return hashlib.md5(text.encode()).hexdigest()
