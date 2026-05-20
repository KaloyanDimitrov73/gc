from collections import deque
from typing import List, Optional
from typing_extensions import override
from dataclasses import dataclass

from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.core.language_model import LLMAdapter
from sqa_system.core.data.models import QAPair, Subgraph, Triple
from sqa_system.core.logging.logging import get_logger
from sqa_system.knowledge_base.knowledge_graph.storage import PathBuilder

from ...base.clustering_strategy import (
    ClusteringStrategy, ClusterStrategyOptions, GenerationOptions, ClusterInformation)
from ...utils.cluster_builder import ClusterBuilder

logger = get_logger(__name__)


@dataclass
class AdditionalInformationRestriction:
    """
    This class represents an additional information restriction for the
    MultipleInformationQuestionGenerator. It is used to filter the information
    that is used to generate the questions.

    Args:
        information_predicate (str): The predicate used to filter the information.
        information_value_restriction (Optional[str]): The value that the restriction
            must have to be included in the question generation.
        information_value_predicate_restriction (Optional[str]): Only collects those 
            values where the predicate is equal to this value.
        split_clusters (bool): If set to true, the cluster information
            is split by the restriction. This means that for each value of the restriction
            a new cluster is created. If set to false, the cluster information is not split.
            This means that all values are added into the cluster.
    """
    information_predicate: str = "publication year"
    information_value_restriction: Optional[str | List[str]] = None
    information_value_predicate_restriction: Optional[str | List[str]] = None
    split_clusters: bool = False


@dataclass
class ClusterGeneratorOptions:
    """
    Options for the MultipleInformationQuestionGenerator.

    Args:
        generation_options (GenerationOptions): The options for the generation.
        additional_restrictions (List[AdditionalInformationRestriction], optional): A list of
            additional restrictions that can be used to filter the information used for
            question generation.
        only_use_cluster_with_most_triples (bool): If set to true, the cluster with the most
            triples is used for the question generation. Has priority over the least option.
        only_use_cluster_with_least_triples (bool): If set to true, the cluster with the least
            triples is used for the question generation.
    """
    generation_options: GenerationOptions
    additional_restrictions: List[AdditionalInformationRestriction] = None
    only_use_cluster_with_most_triples: bool = False
    only_use_cluster_with_least_triples: bool = False


class ClusterBasedQuestionGenerator(ClusteringStrategy):
    """
    This generator generates listing based QA pairs

    Args:
        graph (Graph): The graph to generate the QA pairs from.
        llm_adapter (LLMAdapter): The LLM adapter to use for the generation.
        cluster_options (ClusterStrategyOptions): The options for the clustering strategy.
        generator_options (ClusterGeneratorOptions): The options for the
            cluster based generator.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 llm_adapter: LLMAdapter,
                 cluster_options: ClusterStrategyOptions,
                 generator_options: ClusterGeneratorOptions):
        super().__init__(graph, llm_adapter, cluster_options,
                         generator_options.generation_options)
        self.generator_options = generator_options

    @override
    def generate(self) -> List[QAPair]:
        """
        The main generator function. It generates the QA pairs based on the
        clustering strategy. It first collects the restrictions subgraphs and
        then proceeds to cluster the information by the similarity of the restriction
        values. The clusters are then filtered by the additional restrictions
        and the final clusters are used to generate the QA pairs.

        Returns:
            List[QAPair]: The generated QA pairs.
        """

        final_clusters, restriction_subgraphs = self._prepare_clusters()

        logger.info("Using %s clusters for generation", len(final_clusters))

        # Add additional requirements based on this strategy
        self.options.additional_requirements.append(
            "Only generate 1 Question Answer pair!")
        self.options.additional_requirements.append(
            ("The context you are given contains the entities that are asked for. "
             "The list is complete meaning there are no more entities that are not in the list."))
        self.options.additional_requirements.append(
            ("This means you can safely generate the questions and answers"
             " with the provided contexts as its truthfulness is already ensured"))
        self.options.additional_requirements.append(
            "Include all entities from the context in the question and answer generation. "
            "This is important! Even include contexts if you think they are not relevant. ")

        if len(final_clusters) == 0:
            logger.info("No clusters found that match the restrictions.")
            return []

        if self.generator_options.only_use_cluster_with_most_triples:
            final_clusters = {
                1: max(final_clusters.items(), key=lambda x: len(x[1]))[1]}
        elif self.generator_options.only_use_cluster_with_least_triples:
            final_clusters = {
                1: min(final_clusters.items(), key=lambda x: len(x[1]))[1]}

        qa_pairs = self._run_qa_generation_for_clusters(
            clusters=final_clusters,
            restriction_subgraphs=restriction_subgraphs
        )
        return qa_pairs

    def _prepare_clusters(self) -> tuple[dict[int, list[ClusterInformation]], dict]:
        """
        Helper function to prepare the publication clusters.

        Returns:
            tuple[dict[int, list[ClusterInformation]], dict]: A tuple containing the clusters
                and the restriction subgraphs.            
        """

        restriction_subgraphs = self._prepare_restriction_subgraphs()

        restrictions = self._collect_restrictions(restriction_subgraphs)

        if not self.cluster_options.skip_similarity_clustering:
            clusters = self.cluster_builder.cluster_by_value_similarity(
                cluster_information=restrictions,
                clustering_eps=self.cluster_options.cluster_eps,
                clustering_metric=self.cluster_options.cluster_metric,
                embedding_model=self.llm_embedding_adapter
            )
        else:
            clusters = {1: restrictions}

        logger.info("Found %s inital clusters", len(clusters))
        filtered_clusters = []
        final_clusters = {}
        if (self.generator_options.additional_restrictions is not None and
                len(self.generator_options.additional_restrictions) > 0):
            for cluster_id, cluster in clusters.items():
                if cluster_id < 0:
                    continue

                filtered_cluster_informations = self._filter_clusters_by_restrictions(
                    cluster=cluster,
                    restriction_subgraphs=restriction_subgraphs
                )
                filtered_clusters.extend(filtered_cluster_informations)
            if len(filtered_clusters) > 0:
                final_clusters = {i: filtered_clusters[i]
                                  for i in range(len(filtered_clusters))}
        else:
            final_clusters = clusters

        return final_clusters, restriction_subgraphs

    def _filter_clusters_by_restrictions(self,
                                         cluster: List[ClusterInformation],
                                         restriction_subgraphs: dict) -> list[list[ClusterInformation]]:
        """
        The main loop for adding additional triples to the clusters based on the
        restriction parameters given.

        Args:
            cluster (List[ClusterInformation]): The cluster to filter.
            restriction_subgraphs (dict): The restriction subgraphs to use for filtering.
        Returns:
            list[list[ClusterInformation]]: The filtered clusters.
        """
        clusters_for_next_loop: list[list[ClusterInformation]] = [cluster]
        for option in self.generator_options.additional_restrictions:
            preliminary_clusters: deque[list[ClusterInformation]] = deque()
            preliminary_clusters.extend(clusters_for_next_loop)
            clusters_for_next_loop = []
            while preliminary_clusters:

                updated_cluster_informations = self._retrieve_updated_cluster_information(
                    preliminary_clusters=preliminary_clusters,
                    restriction_subgraphs=restriction_subgraphs,
                    option=option
                )

                if len(updated_cluster_informations) == 0:
                    continue

                clusters_for_next_loop.extend(
                    self._add_updated_cluster_information_to_cluster(
                        updated_cluster_informations=updated_cluster_informations,
                        option=option
                    )
                )

        return clusters_for_next_loop

    def _retrieve_updated_cluster_information(self,
                                              preliminary_clusters: deque[list[ClusterInformation]],
                                              restriction_subgraphs: dict,
                                              option: AdditionalInformationRestriction) -> list[dict]:
        """
        This function collects, for each piece of cluster information, additional triples
        based on the provided restriction options. It iterates through the preliminary
        clusters and, for each, calls helper methods to find relevant triples from
        the knowledge graph that match the specified restrictions.

        Args:
            preliminary_clusters (deque): The clusters to process and potentially update.
            restriction_subgraphs (dict): The restriction subgraphs to use for filtering,
                keyed by root entity.
            option (AdditionalInformationRestriction): The restriction options guiding
                the collection of additional triples.
        Returns:
            list[dict]: A list of dictionaries, each containing the original root entity,
                the original cluster information, and the newly found restriction_triples.
        """
        updated_cluster_informations: list[dict] = []
        for cluster_information in preliminary_clusters.popleft():
            if cluster_information.root_entity not in restriction_subgraphs:
                logger.debug("Found no restriction subgraph for entity %s.",
                             cluster_information.root_entity)
                continue

            restriction_subgraph = restriction_subgraphs[cluster_information.root_entity]

            subgraph_restriction_triples = self._get_first_restriction_values(
                restriction_options=option,
                restriction_subgraph=restriction_subgraph,
                starting_triple=cluster_information.triples[-1]
            )
            if len(subgraph_restriction_triples) == 0:
                continue
            updated_cluster_informations.append(
                {
                    "root_entity": cluster_information.root_entity,
                    "cluster_information": cluster_information,
                    "restriction_triples": subgraph_restriction_triples
                }
            )
        return updated_cluster_informations

    def _add_updated_cluster_information_to_cluster(self,
                                                    updated_cluster_informations: list[dict],
                                                    option: AdditionalInformationRestriction) -> list[list[ClusterInformation]]:
        """
        This function decides how the collected restriction triples are added to the current cluster.
        There are two options: 1) merging all triples into the current cluster or 2) splitting
        the cluster for each distinct value of the restriction triples.

        Args:
            updated_cluster_informations (list[dict]): The updated cluster informations,
                containing root entities, original cluster info, and new restriction triples.
            option (AdditionalInformationRestriction): The restriction options, which includes
                the 'split_clusters' flag determining the merging strategy.
        Returns:
            list[list[ClusterInformation]]: A list of new clusters. If not splitting, this will
                be a list containing a single cluster with all new triples merged. If splitting,
                this will be a list of clusters, each corresponding to a distinct value found
                in the restriction triples.
        """
        clusters_for_next_loop: list[list[ClusterInformation]] = []
        if option.split_clusters:
            triple_value_to_root_mapping: dict[str, list[dict]] = {}
            for data in updated_cluster_informations:
                for triple in data["restriction_triples"]:
                    if triple.entity_object.text not in triple_value_to_root_mapping:
                        triple_value_to_root_mapping[triple.entity_object.text] = [
                        ]

                    triple_value_to_root_mapping[triple.entity_object.text].append({
                        "root_entity": data["root_entity"],
                        "cluster_information": data["cluster_information"],
                        "restriction_triple": triple
                    })

            for _, dict_list in triple_value_to_root_mapping.items():
                updated_semantic_clusters: list[ClusterInformation] = [
                ]
                for data in dict_list:
                    new_cluster_info = data["cluster_information"].model_copy(
                        deep=True)
                    new_cluster_info.triples.append(
                        data["restriction_triple"])
                    updated_semantic_clusters.append(new_cluster_info)
                clusters_for_next_loop.append(
                    updated_semantic_clusters)
        else:
            updated_clusters: list[ClusterInformation] = []
            for data in updated_cluster_informations:
                copied = data["cluster_information"].model_copy(
                    deep=True)
                copied.triples.extend(
                    list(data["restriction_triples"]))
                updated_clusters.append(copied)
            clusters_for_next_loop.append(updated_clusters)

        return clusters_for_next_loop

    def _get_first_restriction_values(self,
                                      restriction_options: AdditionalInformationRestriction,
                                      restriction_subgraph: Subgraph,
                                      starting_triple: Triple,
                                      visited_entities=None):
        """
        This function attempts to find triples that match the specified 'information_predicate'
        by exploring the graph. It starts by looking at paths extending from the subject of
        the 'starting_triple'. If no matching triples are found directly, it recursively
        explores triples connected to the subject of the 'starting_triple' (i.e., incoming
        relations to this subject) to find the desired predicate value.

        Args:
            restriction_options (AdditionalInformationRestriction): The restriction options,
                including the predicate to search for and any value restrictions.
            restriction_subgraph (Subgraph): The subgraph within which to confine the search.
            starting_triple (Triple): The triple from which the graph exploration begins.
                The search for restriction values is related to its entity_subject.
            visited_entities (set, optional): A set of visited entities (object of triples)
                to avoid cycles during recursive calls. Defaults to None, initialized if so.

        Returns:
            list[Triple]: A list of restriction triples that were found during the traversal,
                matching the criteria in 'restriction_options'.
        """
        if visited_entities is None:
            visited_entities = set()

        if starting_triple.entity_object in visited_entities:
            return []

        visited_entities.add(starting_triple.entity_object)

        path_builder = PathBuilder(restriction_subgraph)
        cluster_builder = ClusterBuilder(
            graph=self.graph, llm_adapter=self.llm_adapter
        )
        restriction_triples = set()
        for path in path_builder.build_all_paths(
            current=starting_triple.entity_subject,
            include_against_direction=False,
            include_tails=False
        ):
            triples = cluster_builder.get_restriction_triples_from_path(
                path=path,
                restriction=restriction_options.information_predicate,
                check_predicate=True,
                restriction_value=restriction_options.information_value_restriction
            )
            if restriction_options.information_value_predicate_restriction:
                if not isinstance(restriction_options.information_value_predicate_restriction, list):
                    predicate_restrictions = [
                        restriction_options.information_value_predicate_restriction]
                else:
                    predicate_restrictions = restriction_options.information_value_predicate_restriction
                for predicate in predicate_restrictions:
                    filtered_triples = [
                        triple for triple in triples
                        if (triple.predicate.lower() == predicate.lower())
                    ]
                    restriction_triples.update(filtered_triples)
            else:
                restriction_triples.update(triples)
        if restriction_triples:
            return list(restriction_triples)

        head_relations = self.graph.get_relations_of_tail_entity(
            starting_triple.entity_subject)

        for relation in head_relations:
            triples = self._get_first_restriction_values(
                restriction_options=restriction_options,
                restriction_subgraph=restriction_subgraph,
                starting_triple=relation,
                visited_entities=visited_entities
            )
            if triples:
                restriction_triples.update(triples)

        return list(restriction_triples)
