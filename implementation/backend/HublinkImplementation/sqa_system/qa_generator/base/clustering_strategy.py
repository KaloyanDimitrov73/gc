from abc import ABC
import ast
import random
from typing import List, Set, Tuple, Dict, Optional
from pydantic import BaseModel

from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.core.language_model import LLMAdapter
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.config.models import EmbeddingConfig
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.knowledge_base.knowledge_graph.storage.utils.path_builder import PathBuilder
from sqa_system.knowledge_base.knowledge_graph.storage import (
    SubgraphBuilder,
    SubgraphOptions
)
from sqa_system.core.data.models import (
    QAPair,
    Knowledge,
    Triple
)
from sqa_system.qa_generator.question_classifier import QuestionClassifier
from sqa_system.core.logging.logging import get_logger

from ..utils.cluster_builder import ClusterBuilder, ClusterInformation
from ..utils.from_subgraph_generator import FromSubgraphGenerator, SubgraphGeneratorOptions
from .kg_qa_generation_strategy import KGQAGenerationStrategy, GenerationOptions

logger = get_logger(__name__)


class ClusterStrategyOptions(BaseModel):
    """
    Options for the ListingQuestionGenerator.

    Args:
        restriction_type (str): The predicate type used for the initial retrieval
            of entities (e.g., publications) from the knowledge graph. All entities
            containing a triple with this predicate type are selected as the
            initial set. For instance, this could be used to select all publications
            within a specific research field or by a particular author.
        restriction_text (str): A predicate type used to further filter the initially
            retrieved entities. The subgraph associated with each entity is examined;
            if a triple with this predicate type exists, the entity is retained,
            otherwise it is discarded.
        restriction_value (Optional[Union[str, List[str]]]): Specifies the required
            object value(s) for the triple identified by `restriction_text`. If the
            predicate specified by `restriction_text` is found in an entity's subgraph,
            the object value of that triple must match this value (or one of the values
            in the list) for the entity to be kept. If None, only the presence of the
            `restriction_text` predicate is checked. This allows filtering for specific
            values, such as publications by a defined set of authors. Defaults to None.
        cluster_eps (Optional[float]): The epsilon radius parameter for the DBSCAN
            clustering algorithm. Clustering is performed on embeddings derived from
            the values associated with the `restriction_text` predicate within the
            filtered entities. This parameter influences the neighborhood size for
            point density calculations in DBSCAN. Defaults to 0.5. Is only used if
            `skip_similarity_clustering` is False.
        cluster_metric (Optional[str]): The distance metric used by the DBSCAN
            algorithm during clustering. Common options include 'cosine', 'euclidean',
            or 'manhattan'. Defaults to 'cosine'. Is only used if
            `skip_similarity_clustering` is False.
        cluster_emb_config (Optional[EmbeddingConfig]): Configuration object specifying
            the model and parameters for generating embeddings. These embeddings,
            derived from values associated with `restriction_text`, are used as input
            for the DBSCAN clustering algorithm. Defaults to None. Is only used if
            `skip_similarity_clustering` is False.
        golden_triple_limit (int): The maximum number of RDF triples (golden triples)
            allowed to be associated with a single generated question-answer pair.
            Pairs exceeding this limit are discarded, controlling the maximum complexity
            or scope of information per question. Defaults to 99.
        golden_triple_minimum (int): The minimum number of RDF triples (golden triples)
            required for a generated question-answer pair to be considered valid.
            Pairs with fewer triples than this minimum are discarded. This helps filter
            out overly simplistic or uninformative questions. Defaults to 1.
        soft_limit_qa_pairs (int): A target limit for the total number of question-answer
            pairs to generate. The generation process attempts to stop once this number
            is reached. Due to batch processing by the language model, the final count
            might slightly exceed this limit. Defaults to 99.
        topic_entity (Optional[Knowledge]): An optional entity from the knowledge graph
            representing the central topic. While not directly used during the clustering
            or generation phase, this entity is associated with the output question-answer
            pairs to potentially aid downstream retrieval processes. Defaults to None.
        topic_entity_description (Optional[str]): An optional textual description of the
            `topic_entity`. If provided, this description is incorporated into the
            generated question after the initial generation by the language model,
            providing additional context. Defaults to None.
        skip_clusters_with_only_one_root (bool): If True, clusters containing only a
            single root entity (e.g., one publication) after filtering and clustering
            are ignored. This ensures that generated questions are based on patterns
            found across multiple related entities, rather than isolated instances.
            Defaults to True.
        skip_similarity_clustering (bool): If True, the DBSCAN similarity clustering step
            (based on `cluster_eps`, `cluster_metric`, `cluster_emb_config`) is bypassed.
            Questions are generated directly from the groups formed by the initial
            `restriction_type`, `restriction_text`, and `restriction_value` filtering.
            Defaults to False.
        enable_caching (bool): If True, enables caching of the subgraphs retrieved
            during the entity selection process. This can significantly speed up
            subsequent runs with the same or overlapping restrictions. Defaults to False.
        use_predicate_as_value (bool): If True, utilizes the predicate text itself as
            the significant value instead of the object value from an RDF triple.
            This is useful in cases where the object value is non-informative (e.g.,
            a boolean literal like `True` in `(paper, hasUsedGuidelines, True)`),
            and the predicate carries more semantic meaning for clustering or question
            generation. Defaults to False.
        limit_restrictions (Optional[int]): An optional upper limit on the number of
            relevant triples (restrictions, which become golden triples) within a
            single cluster *before* question generation. If a cluster contains more
            triples matching the restriction criteria than this limit, the entire
            cluster is skipped. This prevents generating questions from excessively
            large or dense clusters. Defaults to None. 
    """
    restriction_type: str
    restriction_text: str
    restriction_value: Optional[str | List[str]] = None
    cluster_eps: Optional[float] = 0.5
    cluster_metric: Optional[str] = None
    cluster_emb_config: Optional[EmbeddingConfig] = None
    golden_triple_limit: int = 99
    golden_triple_minimum: int = 1
    soft_limit_qa_pairs: int = 99
    topic_entity: Optional[Knowledge] = None
    topic_entity_description: Optional[str] = None
    skip_clusters_with_only_one_root: bool = True
    skip_similarity_clustering: bool = False
    enable_caching: bool = False
    use_predicate_as_value: bool = False
    limit_restrictions: Optional[int] = None


class InternalOptions(BaseModel):
    """
    Additional internal options for controlling the behavior of the clustering strategy.

    Args:
        classify_qa_pairs (bool): If True, generated QA pairs will be classified.
        check_if_all_cluster_triples_are_in_generated (bool): If True, ensures that all
            triples from the original cluster are included in the golden_triples of the
            generated QA pairs.
    """
    classify_qa_pairs: bool = True
    check_if_all_cluster_triples_are_in_generated: bool = True


class ClusteringStrategy(KGQAGenerationStrategy, ABC):
    """
    A base class for QA generation strategies that utilize clustering techniques.

    This class provides a framework for generating question-answer pairs by first
    identifying clusters of related information within a knowledge graph and then
    generating questions based on these clusters.

    Args:
        graph: The knowledge graph instance to use for QA generation.
        llm_adapter: The language model adapter for interacting with an LLM.
        cluster_options: Configuration options specific to the clustering process.
        options: General configuration options for QA generation.
    """

    _internal_options: InternalOptions = InternalOptions()

    def __init__(self,
                 graph: KnowledgeGraph,
                 llm_adapter: LLMAdapter,
                 cluster_options: ClusterStrategyOptions,
                 options: GenerationOptions):
        super().__init__(graph, llm_adapter, options)
        self.cluster_options = cluster_options
        self.cluster_builder = ClusterBuilder(
            graph=self.graph, llm_adapter=self.llm_adapter)
        self.subgraph_generator = FromSubgraphGenerator(
            graph=self.graph, llm_adapter=self.llm_adapter)
        self.subgraph_builder = SubgraphBuilder(knowledge_graph=self.graph)
        self.progress_handler = ProgressHandler()
        self.question_classifier = QuestionClassifier(self.llm_adapter)
        self.llm_embedding_adapter = None
        if self.cluster_options and self.cluster_options.cluster_emb_config:
            self.llm_embedding_adapter = LLMProvider().get_embeddings(
                embedding_config=self.cluster_options.cluster_emb_config)

    def _prepare_restriction_subgraphs(self) -> Dict[str, List[Triple]]:
        """
        Prepares subgraphs that contain the restrictions for the clustering.
        It retrieves the restriction entities from the graph and generates
        subgraphs for each restriction entity. 

        Returns:
            A dictionary containing the restriction subgraphs.        
        """
        restriction_entities = list(self.graph.get_entities_by_predicate_id(
            [self.cluster_options.restriction_type]))
        if not restriction_entities:
            logger.info("No restriction entities found. " +
                        "Skipping preparation of restriction subgraphs.")
            return {}

        # Add a task to the progress handler
        task_id = self.progress_handler.add_task(
            "restriction_subgraphs",
            "Preparing restriction subgraphs...",
            total=len(restriction_entities)
        )
        if self.cluster_options.topic_entity:
            restriction_subgraphs, _ = self.subgraph_builder.get_subgraphs_that_include_entity(
                root_entities=restriction_entities,
                entity_to_include=self.cluster_options.topic_entity,
                use_caching=self.cluster_options.enable_caching
            )
            self.progress_handler.update_task_by_string_id(
                task_id, advance=len(restriction_entities))
        else:
            restriction_subgraphs = {}
            for root_entity in restriction_entities:
                root, restriction_subgraph = self.subgraph_builder.get_subgraph(
                    root_entity=root_entity,
                    options=SubgraphOptions(
                        hop_amount=-1,
                        size_limit=-1,
                        go_against_direction=True,
                    ),
                    use_caching=self.cluster_options.enable_caching
                )
                restriction_subgraphs[root.uid] = restriction_subgraph
                self.progress_handler.update_task_by_string_id(
                    task_id, advance=1)
        self.progress_handler.finish_by_string_id(task_id)
        return restriction_subgraphs

    def _run_qa_generation_for_clusters(
            self,
            clusters: Dict[int, List[ClusterInformation]],
            restriction_subgraphs: Dict[str, List[Triple]]) -> List[QAPair]:
        """
        Runs the QA generation for the clusters. It iterates over the clusters
        and generates QA pairs for each cluster. The clusters are shuffled to
        get a random order. The function stops when the soft limit of QA pairs
        is reached. The function also checks if the clusters are valid and
        generates the QA pairs only for valid clusters.

        Args:
            clusters: The clusters to generate QA pairs for.
            restriction_subgraphs: The restriction subgraphs to use for the
                generation of the QA pairs.

        Returns:
            A list of generated QA pairs.
        """
        qa_pairs = []
        prepared_clusters = []
        for cluster_id, cluster in clusters.items():
            # The negative cluster contains the elements without
            # similarities to other elements. We skip it.
            if cluster_id < 0:
                continue
            prepared_clusters.append(cluster)

        # shuffle the clusters to get a random order
        random.shuffle(prepared_clusters)

        for cluster in prepared_clusters:
            if len(qa_pairs) >= self.cluster_options.soft_limit_qa_pairs:
                logger.info(
                    f"Reached the soft limit of {self.cluster_options.soft_limit_qa_pairs}.")
                break

            generated_pairs = self._run_qa_generation_for_a_cluster(
                cluster=cluster,
                restriction_subgraphs=restriction_subgraphs
            )

            qa_pairs.extend(generated_pairs)
        return qa_pairs

    def _run_qa_generation_for_a_cluster(self,
                                         cluster: List[ClusterInformation],
                                         restriction_subgraphs: Dict[str, List[Triple]]) -> List[QAPair]:
        """
        Runs the QA generation for a single cluster. It generates the QA pairs
        for the cluster and checks if the generated pairs are valid. 

        Args:
            cluster: The cluster to generate QA pairs for.
            restriction_subgraphs: The restriction subgraphs to use for the
                generation of the QA pairs.

        Returns:
            A list of generated QA pairs.
        """
        # We also check if the information all comes from one
        # root entity. If so, we skip it depending on the configuration
        if self.cluster_options.skip_clusters_with_only_one_root:
            root_entities = set()
            for information in cluster:
                root_entities.add(information.root_entity)
            if len(root_entities) < 2:
                return []

        # Shuffle the data inside of the cluster to get more randomized results
        random.shuffle(cluster)

        # Prepare the additional requirements text
        requirements_text = None
        if self.options.additional_requirements:
            requirements_text = ""
            for i, req in enumerate(self.options.additional_requirements):
                requirements_text += f"{i + 1}. {req}\n"

        golden_triples = []
        for information in cluster:
            golden_triples.extend(information.triples.copy())

        # Generate the QA-Pairs
        generated_pairs, _ = self.subgraph_generator.generate_from_subgraph(
            SubgraphGeneratorOptions(
                root_entity=self.cluster_options.topic_entity,
                subgraph=golden_triples,
                template_text=self.options.template_text,
                additional_requirements=requirements_text,
                strategy_name="clustering_strategy",
                validate_contexts=self.options.validate_contexts,
                classify_qa_pairs=False,
                convert_path_to_text=self.options.convert_path_to_text
            )
        )

        filtered_pairs = self._check_if_pairs_are_within_constraints(
            generated_pairs=generated_pairs
        )
        if not filtered_pairs:
            logger.info("No valid pairs found. Skipping cluster.")
            return []

        validated_pairs = self._validate_generated_pairs(
            generated_pairs=filtered_pairs,
            golden_triples=golden_triples,
            cluster=cluster,
            restriction_subgraphs=restriction_subgraphs
        )

        updated_pairs = []
        classify_qa_pairs = False
        if self.options.classify_questions:
            classify_qa_pairs = self._internal_options.classify_qa_pairs
        if classify_qa_pairs:
            for pair in validated_pairs:
                pair = self.question_classifier.classify_qa_pair(
                    qa_pair=pair,
                )
                updated_pairs.append(pair)
        else:
            return validated_pairs
        return updated_pairs

    def _check_if_pairs_are_within_constraints(self, generated_pairs: List[QAPair]) -> List[QAPair]:
        """
        Verifies if the generated pairs are within the constraints of the
        clustering strategy. 

        Args:
            generated_pairs: The generated pairs to check.
        Returns:
            A list of valid pairs.
        """
        valid_pairs = []
        for pair in generated_pairs:
            golden_triples = pair.golden_triples
            if len(golden_triples) > self.cluster_options.golden_triple_limit:
                logger.info((f"The generated pair has a golden triple size of {len(golden_triples)} which "
                            f"is higher than the limit of {self.cluster_options.golden_triple_limit}."))
                continue

            if len(golden_triples) < self.cluster_options.golden_triple_minimum:
                logger.info((f"The generated pair has a golden triple size of {len(golden_triples)} which "
                            f"is lower than the minimum of {self.cluster_options.golden_triple_minimum}."))
                continue
            valid_pairs.append(pair)
        return valid_pairs

    def _validate_generated_pairs(self,
                                  generated_pairs: List[QAPair],
                                  golden_triples: List[Triple],
                                  cluster: List[ClusterInformation],
                                  restriction_subgraphs: Dict[str, List[Triple]]) -> List[QAPair]:
        """
        Validates the generated pairs by checking if the golden triples are
        present in the generated pairs. 

        Args:
            generated_pairs: The generated pairs to validate.
            golden_triples: The golden triples to check.
            cluster: The cluster to check.
            restriction_subgraphs: The restriction subgraphs to use for the
                validation of the generated pairs.

        Returns:
            A list of valid generated pairs.
        """
        if self._internal_options.check_if_all_cluster_triples_are_in_generated:
            for pair in generated_pairs:
                pair.golden_triples = []
                pair.source_ids = []
                for triple in golden_triples:
                    pair.golden_triples.append(str(triple))
                    paper_entity = self.graph.get_paper_from_entity(
                        triple.entity_subject)
                    relations = self.graph.get_relations_of_head_entity(
                        paper_entity)
                    source_id = ""
                    for relation in relations:
                        if "doi" in relation.predicate:
                            source_id = relation.entity_object.text
                            break
                    pair.source_ids.append(source_id)

        complete_subgraph = []
        for information in cluster:
            information_subgraph = restriction_subgraphs[information.root_entity]
            complete_subgraph.extend(information_subgraph)

        for pair in generated_pairs:
            # Calculate the hop amount
            if self.cluster_options.topic_entity:
                calculated_hop_amount = self.calculate_hop_amount(
                    root_entity=self.cluster_options.topic_entity,
                    subgraph=complete_subgraph,
                    golden_triples=pair.golden_triples
                )
                if calculated_hop_amount <= 0:
                    logger.info(
                        f"Skipping QA-Pair with invalid hop amount of {calculated_hop_amount}")
                    continue
                pair.hops = calculated_hop_amount

            # Add topic entity restriction
            if self.cluster_options.topic_entity:
                if self.cluster_options.topic_entity_description:
                    pair.question = self.add_information_to_question(
                        question=pair.question,
                        information=self.cluster_options.topic_entity_description
                    )
        return generated_pairs

    def _collect_restrictions(self,
                              restriction_subgraphs: dict) -> List[ClusterInformation]:
        """
        Extracts the restrictions from the provided subgraphs. The function
        processes the paths in the subgraphs and for each root entity, it
        creates a ClusterInformation object.

        Args:
            restriction_subgraphs: The subgraphs containing the restrictions.

        Returns:
            A list of ClusterInformation objects.
        """
        restrictions = []
        seen_combinations = set()
        for root, subgraph in restriction_subgraphs.items():
            if self.cluster_options.limit_restrictions:
                if len(restrictions) >= self.cluster_options.limit_restrictions:
                    logger.debug(
                        f"Reached the limit of {self.cluster_options.limit_restrictions} restrictions.")
                    break
            builder = PathBuilder(subgraph)
            paths = builder.build_all_paths(
                current=root,
                include_tails=True,
                include_against_direction=True
            )
            for path in paths:
                restrictions.extend(self._process_path(
                    root=root,
                    path=path,
                    seen_combinations=seen_combinations
                ))
        return restrictions

    def _process_path(self,
                      root: str,
                      path: List[Triple],
                      seen_combinations: Set[tuple]) -> List[ClusterInformation]:
        """
        Processes a path to extract restriction triples and generates corresponding cluster
        information.

        This function retrieves restriction triples from the provided path based on the specified
        restriction type. Each unique combination of the root entity and the restriction entity
        is used to create a new ClusterInformation object.
        Combinations encountered more than once are not processed again.

        Args:
            root: The root entity of the path.
            path: The path containing the restriction triples.
            seen_combinations: A set to track unique combinations of root and restriction entities.

        Returns:
            A list of ClusterInformation objects created from the restriction triples.
        """
        restrictions = []
        restriction_values = self.cluster_builder.get_restriction_triples_from_path(
            path=path,
            restriction=self.cluster_options.restriction_text,
            check_predicate=True,
            restriction_value=self.cluster_options.restriction_value
        )
        for restriction in restriction_values:
            if self.cluster_options.use_predicate_as_value:
                raw_value = restriction.predicate
            else:
                raw_value = restriction.entity_object.text

            # Check if the value is a list of values as string
            # if so, we split it and add each value as a restriction
            if raw_value.startswith('[') and raw_value.endswith(']'):
                values = ast.literal_eval(str(raw_value))
                if not isinstance(values, list):
                    values = [raw_value]
            else:
                values = [raw_value]

            for value in values:
                # Create tuple of current combination
                combination = (root, str(value))
                # Only add if not seen before
                if str(value) == "True" or str(value) == "False" or combination not in seen_combinations:
                    seen_combinations.add(combination)
                    description, context_triple = self._get_relation_description_or_default(
                        restriction, value)
                    # This line is a workaround for a validation issue that occurs when
                    # the context triple is loaded from a serzialized object.
                    # If we remove this line it throws an error from the pydantic library
                    # which we dont know why this is happening.
                    context_triple = Triple.model_validate_json(
                        context_triple.model_dump_json())
                    restrictions.append(ClusterInformation(
                        root_entity=root,
                        value=description,
                        triples=[context_triple]
                    ))
        return restrictions

    def _get_relation_description_or_default(self,
                                             restriction: Triple,
                                             default: str) -> Tuple[str, Triple]:
        """
        Retrieves the description of a relation or returns a default value if not found.

        Args:
            restriction: The restriction triple to check.
            default: The default value to return if no description is found.

        Returns:
            A tuple containing the description value and the context triple.
        """
        relations = self.graph.get_relations_of_head_entity(
            restriction.entity_object)
        description = None
        context_triple = None
        for relation in relations:
            if "description" in relation.predicate.lower():
                description = relation.entity_object.text
                context_triple = relation
                break
        if description:
            value = description
        else:
            context_triple = restriction
            value = default
        return value, context_triple
