from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.cluster import DBSCAN

from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.language_model import EmbeddingAdapter
from sqa_system.qa_generator.question_classifier import QuestionClassifier
from sqa_system.knowledge_base.knowledge_graph.storage.utils.subgraph_builder import SubgraphBuilder
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models import Triple

from sqa_system.knowledge_base.knowledge_graph.storage.utils import (
    GraphPathFilter
)
from sqa_system.core.data.models.knowledge import Knowledge
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


class ClusterInformation(BaseModel):
    """
    Information used for clustering by similarity.
    """
    root_entity: Knowledge = Field(...,
                                   description='The root entity of the cluster')
    value: str = Field(..., description='The value of the cluster')
    triples: List[Triple] = Field(...,
                                  description='The triples of the cluster')


class ClusterBuilder:
    """
    The ClusterBuilder class is responsible for building clusterings from subgraphs.

    Args:
        graph (KnowledgeGraph): The knowledge graph to be used for clustering.
        llm_adapter (LLMAdapter): The language model adapter to be used for clustering.
    """

    def __init__(self, graph: KnowledgeGraph, llm_adapter: LLMAdapter):
        self.graph = graph
        self.llm_adapter = llm_adapter
        self.progress_handler = ProgressHandler()
        self.subgraph_builder = SubgraphBuilder(self.graph)
        self.path_filter = GraphPathFilter()
        self.question_classifier = QuestionClassifier(llm_adapter=llm_adapter)

    def cluster_by_value_similarity(
            self,
            cluster_information: List[ClusterInformation],
            embedding_model: EmbeddingAdapter,
            clustering_eps: float = 0.3,
            clustering_metric: str = 'cosine') -> Dict[int, List[ClusterInformation]]:
        """
        Clusters the provided cluster information based on the value similarity
        using the provided embedding model with the DBSCAN algorithm.

        Args:
            cluster_information (List[ClusterInformation]): A list of ClusterInformation 
                instances containing the information to be clustered.
            embedding_model (EmbeddingAdapter): The embedding model used to embed the values.
            clustering_eps (float): The epsilon value for the clustering.
            clustering_metric (str): The metric for the clustering.

        Returns:
            Dict[int, List[ClusterInformation]]: A dictionary mapping cluster labels to
                lists of ClusterInformation instances. The keys are the cluster labels,
                and the values are lists of ClusterInformation instances that belong to
                that cluster.
        """
        if len(cluster_information) == 0:
            return {}
        df = pd.DataFrame([vars(r) for r in cluster_information])
        embeddings = embedding_model.embed_batch(df["value"].tolist())

        logger.debug("Running DBSCAN clustering")
        dbscan = DBSCAN(eps=clustering_eps, min_samples=2,
                        metric=clustering_metric)
        cluster_labels = dbscan.fit_predict(embeddings)
        df['cluster'] = cluster_labels

        # Create cluster mapping
        mapping = {}
        for _, row in df.iterrows():
            cluster = row['cluster']
            if cluster not in mapping:
                mapping[cluster] = []
            mapping[cluster].append(ClusterInformation(
                root_entity=row['root_entity'],
                value=row['value'],
                triples=row['triples']
            ))
        return mapping

    def cluster_paths_containing_entity_connected_to_entity(self,
                                                            all_paths: List[List[Triple]],
                                                            entity_substring: str,
                                                            root_entity: Knowledge) -> dict:
        """
        Filters and clusters entity paths based on a specified golden entity, 
        ensuring each cluster is connected to the root entity.

        Args:
            all_paths (List[List[Triple]]): A list of all possible paths, 
                where each path is a list of Triple instances.
            entity_substring (str): The entity substring to cluster the paths by.
            root_entity (Knowledge): The root entity to verify connectivity 
                within the clustered paths.

        Returns:
            dict: A dictionary mapping the golden entity to their associated paths.
                It has the following structure:
                {
                    "uid": [List[Triple], List[Triple], ...],
                    ...
                }

        """
        golden_entity_paths = self.cluster_paths_by_id_substring(
            all_paths, entity_substring)
        # Next we check each of the golden entity paths if they are connected to the root entity
        # If no direct path is found that contains the root entity we remove the golden entity
        # completely
        golden_entities_to_remove = []
        for golden_entity_id, paths in golden_entity_paths.items():
            found_root_entity = False
            for path in paths:
                for triple in path:
                    if root_entity.uid in (triple.entity_subject.uid, triple.entity_object.uid):
                        found_root_entity = True
                        break
                if found_root_entity:
                    break
            if not found_root_entity:
                golden_entities_to_remove.append(golden_entity_id)

        for entity in golden_entities_to_remove:
            golden_entity_paths.pop(entity, None)
        return golden_entity_paths

    def cluster_paths_by_id_substring(self,
                                      all_paths: List[List[Triple]],
                                      id_substring: str) -> dict:
        """
        This method iterates through all provided paths and groups them according to 
        whether the id appears within the paths.

        Args:
            all_paths (List[List[Triple]]): A list of paths, where each path is a list of 
                Triple instances.
            id (str): The id substring to cluster the paths by.

        Returns:
            dict: A dictionary mapping the uid to their associated paths.

        The returning dictionary has the following structure:
        {
            "uid": [List[Triple], List[Triple], ...],
            ...
        }
        """
        golden_entity_paths = {}
        # First we cluster the paths by their golden entity
        for path in all_paths:
            for triple in path:
                if id_substring in triple.entity_object.uid:
                    if golden_entity_paths.get(triple.entity_object.uid, None) is None:
                        golden_entity_paths[triple.entity_object.uid] = []
                    golden_entity_paths[triple.entity_object.uid].append(path)
                    break
                if id_substring in triple.entity_subject.uid:
                    if golden_entity_paths.get(triple.entity_subject.uid, None) is None:
                        golden_entity_paths[triple.entity_subject.uid] = []
                    golden_entity_paths[triple.entity_subject.uid].append(path)
                    break
        return golden_entity_paths

    def build_restriction_to_golden_path_mapping(self,
                                                 golden_entity_paths: dict,
                                                 restriction: str,
                                                 golden_entity_substring: str) -> dict:
        """
        Builds a mapping of restriction values to their corresponding golden triples and paths.
        This method processes the provided `golden_entity_paths` to construct a dictionary
        where each key is a restriction value, and the value is a list of mappings containing
        the associated golden triple and the paths leading to it. 

        Args:
            golden_entity_paths (dict): A dictionary mapping golden entity identifiers to their
                associated paths. Expected to have the following structure:
                {
                    "uid": [List[Triple], List[Triple], ...],
                    ...
                }
            restriction (str): The predicate string used to filter and identify relevant triples
                within each path.
            golden_entity_substring (str): The substring used to identify golden entities within 
                the paths.

        Returns:
            dict: A dictionary mapping restriction values to their corresponding golden triples
                and paths. Has the following structure:
        {
            "restriction_value": [
                {
                    "golden_triple": Triple,
                    "path": List[List[Triple]]
                },
                ...
            ],
            ...
        }
        """

        task_id = self.progress_handler.add_task(
            description="Building Clusters..",
            string_id="building_clusters",
            total=len(golden_entity_paths)
        )
        restriction_mapping = {}
        for index, (restriction_value, paths) in enumerate(golden_entity_paths.items()):
            logger.debug(
                f"Processing Golden Entity Paths: {index}|{len(golden_entity_paths)}")
            for path in paths:
                restriction_triples: List[Triple] = self.get_restriction_triples_from_path(
                    path, restriction)
                golden_triple = self._get_golden_triple_from_path(
                    path, golden_entity_substring)

                # If the path doesnt include both the restriction and the golden entity
                # we skip it
                if len(restriction_triples) == 0 or golden_triple is None:
                    continue

                # Check if we have a mapping for the restriction value
                # if not we create a new one
                for restriction_triple in restriction_triples:
                    restriction_value = self.graph.get_entity_by_id(
                        restriction_triple.entity_object.uid).text
                    if restriction_mapping.get(restriction_value, None) is None:
                        restriction_mapping[restriction_value] = []

                    # Now we add the path to the mapping
                    golden_entity_already_added = False
                    for mapping in restriction_mapping[restriction_value]:
                        if mapping.get("golden_triple") == golden_triple:
                            mapping["path"].append(path)
                            golden_entity_already_added = True
                            break
                    if not golden_entity_already_added:
                        restriction_mapping[restriction_value].append(
                            {
                                "golden_triple": golden_triple,
                                "path": [path]
                            }
                        )
            self.progress_handler.update_task_by_string_id(task_id, 1)
        self.progress_handler.finish_by_string_id(task_id)
        return restriction_mapping

    def calculate_hop_amount_from_cluster(self,
                                          mapping: dict,
                                          restriction_value: str) -> int:
        """
        Calculates the amount of hops required for the provided mapping.
        This method iterates through the mapping and determines the maximum number of hops
        needed to reach the specified restriction value from the golden triples.

        Args:
            mapping (dict): A dictionary mapping restriction values to their corresponding
                golden triples and paths. Expected to have the following structure:
                {
                    "restriction_value": [
                        {
                            "golden_triple": Triple,
                            "path": List[List[Triple]]
                        },
                        ...
                    ],
                    ...
                }
            restriction_value (str): The restriction value to calculate the hop amount for.
        Returns:
            int: The maximum number of hops required to reach the specified restriction value
                from the golden triples. If no paths are found, returns -1.
        """
        hop_amount = -1
        for entry in mapping:
            entry_hops = 99999
            for path in entry["path"]:
                hops = 1
                for triple in path:
                    if restriction_value in triple.predicate:
                        break
                hops += 1
            entry_hops = min(entry_hops, hops)
            hop_amount = max(hop_amount, entry_hops)
        return hop_amount

    def _get_golden_triple_from_path(self,
                                     path: List[Triple],
                                     golden_entity_substring: str) -> Triple:
        """
        Retrieves the golden triple from a given path based on a specified substring.
        This method checks each triple in the path to see if the substring is present
        in either the entity object or the entity subject of the triple.

        Args:
            path (List[Triple]): The list of triples representing the path to be analyzed.
            golden_entity_substring (str): The substring used to identify the golden entity
                within the triples.
        Returns:
            Triple: The first triple in the path that contains the specified substring
                in either the entity object or the entity subject. If no such triple is found,
                returns None.
        """
        for triple in path:
            if (golden_entity_substring in triple.entity_object.uid or
                    golden_entity_substring in triple.entity_subject.uid):
                return triple
        return None

    def get_restriction_triples_from_path(self,
                                          path: List[Triple],
                                          restriction: str,
                                          restriction_value: Optional[str |
                                                                      List[str]] = None,
                                          check_predicate: bool = True) -> List[Triple]:
        """
        Extracts triples with literal objects based on a restriction found in a path.

        This method iterates through the input `path`. For the first triple that matches
        the `restriction` (either by its predicate or entity types, depending on
        `check_predicate`), it identifies associated triples whose objects are literals.

        If the matching triple's object is already a literal (not a valid URI/ID, as
        determined by `self.graph.is_valid_id`), that triple itself is considered
        (if it passes the `restriction_value` check). If the matching triple's object
        is a URI/ID, the method calls `_find_restriction_values` to find connected
        triples whose objects are literals.

        The search stops after the first triple in the `path` that satisfies the
        `restriction` criteria and its associated literal-object triples are processed.

        Args:
            path (List[Triple]): The list of triples representing the path to be analyzed.
            restriction (str): The predicate string used to identify restrictive triples 
                within the path.
            restriction_value (Optional[str]): The value that the restriction has to
                have. If not provided, it will be ignored.
            check_predicate (bool): A flag indicating whether to check the predicate of the
                triple for the restriction. If false, it checks the type.
        Returns:
            List[Triple]: A list of triples that match the specified restriction predicate
                and contain the specified restriction value. If no such triples are found,
                returns an empty list.
        """
        restriction_triples: set[Triple] = set()
        for triple in path:
            # Check if the triple contains a label or a type of that restriction
            is_valid = False
            if check_predicate and restriction == triple.predicate:
                is_valid = True
            else:
                subject_types = self.graph.get_types_of_entity(
                    triple.entity_subject)
                object_types = self.graph.get_types_of_entity(
                    triple.entity_object)
                if restriction in subject_types or restriction in object_types:
                    is_valid = True

            if is_valid:
                # We found a subpath that is interesting for us
                # Now we need to find the value(s) to the restriction entity
                if self.graph.is_intermediate_id(triple.entity_object.uid):
                    # If the value is a valid id we need to further traverse
                    # the graph to get the labels
                    triples = self._find_restriction_values(triple)
                    for t in triples:
                        if (restriction_value is not None and
                                not self._check_triple_values(t, restriction_value)):
                            continue
                        restriction_triples.add(t)
                else:
                    if restriction_value is not None:
                        if not self._check_triple_values(triple, restriction_value):
                            continue
                    restriction_triples.add(triple)
                break

        return list(restriction_triples)

    def _find_restriction_values(self, triple: Triple) -> List[Triple]:
        """
        Finds the restriction values for a given triple.
        This method retrieves all triples from the end of the path that are not valid
        identifiers (i.e., do not contain "http://") and returns them as a list.

        Args:
            triple (Triple): The triple for which to find the restriction values.
        Returns:
            List[Triple]: A list of triples that represent the restriction values
                for the given triple. If no such triples are found, returns an empty list.
        """
        end_values = []
        end_triples = self.graph.get_all_triples_from_the_end_of_the_path(
            triple.entity_object)
        if end_triples and len(end_triples) > 0:
            for end_triple in end_triples:
                if "http://" not in end_triple.entity_object.uid:
                    end_values.append(end_triple)
        return end_values

    def _check_triple_values(self,
                             triple: Triple,
                             restriction_value: str | List[str]) -> bool:
        """
        Checks if the triple has the same value as the restriction value.

        Args:
            triple (Triple): The triple to check.
            restriction_value (str | List[str]): The restriction value to compare against.
        Returns:
            bool: True if the triple has the same value as the restriction value,
                False otherwise.
        """
        if isinstance(restriction_value, list):
            for value in restriction_value:
                if triple.entity_object.text == value:
                    return True
        if triple.entity_object.text == restriction_value:
            return True
        return False
