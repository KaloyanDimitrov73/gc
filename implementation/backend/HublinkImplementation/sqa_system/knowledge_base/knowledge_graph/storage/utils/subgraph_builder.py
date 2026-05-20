from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
from collections import deque
import random

from sqa_system.core.data.cache_manager import CacheManager
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.data.models import (
    Triple,
    Knowledge,
    Subgraph
)
from sqa_system.core.logging.logging import get_logger
from ..base.knowledge_graph import KnowledgeGraph
from .path_builder import PathBuilder

logger = get_logger(__name__)


@dataclass
class SubgraphOptions:
    """A class to configure subgraph extraction options.

    This class defines parameters to control how subgraphs are extracted from a larger graph.

    Attributes:
        hop_amount (int): Number of hops to traverse from starting node(s). 
            Default is -1 (unlimited hops).
        size_limit (int): Maximum number of nodes in extracted subgraph.
            Default is -1 (no size limit).
        go_against_direction (bool): Whether to traverse edges against their direction.
            Default is False.
        traverse_all_possible_edges (bool): Whether to explore all possible edges during traversal.
            Default is False.
        filter_types (List[str]): List of types of predicates that filtered for each neighbor
            during traversal. Using the amount_of_filter_types_to_keep parameter, the amount of 
            those predicates to keep can be controlled. Default is an empty list.
        amount_of_filter_types_to_keep (int): Number of relations to keep that have
            the filter types. Default is 1.
    """
    hop_amount: int = -1
    size_limit: int = -1
    go_against_direction: bool = False
    traverse_all_possible_edges: bool = False
    filter_types: List[str] = field(default_factory=list)
    amount_of_filter_types_to_keep: int = 1


class SubgraphBuilder:
    """
    A class that generates subgraphs from a knowledge graph.

    Args:
        knowledge_graph (KnowledgeGraph): The knowledge graph to extract subgraphs from.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self.knowledge_graph = knowledge_graph
        self.progress_handler = ProgressHandler()
        self.cache_manager = CacheManager()
        self.cache_key = f"subgraphs_{knowledge_graph.config.config_hash}"
        self.task = None

    def get_subgraph(
        self,
        root_entity: Optional[Knowledge] = None,
        options: SubgraphOptions = SubgraphOptions(),
        use_caching: bool = False
    ) -> Tuple[Knowledge, Subgraph]:
        """
        Generates a subgraph starting from a root entity.
        If no root entity is provided, a random entity is selected.

        Args:
            root_entity (Optional[Entity]): The root entity to start the subgraph from. If None, 
                a random entity is selected.
            options (SubgraphOptions): Configuration options for subgraph extraction.  
            use_caching (bool): Whether to use caching for the subgraph.
        Returns:
            Tuple[Entity, Subgraph]: A tuple containing the root entity and the generated subgraph.          
        """
        # We select a starting entity
        root_entity = self._select_root_entity(root_entity)

        # We first check whether the subgraph is cached
        if use_caching:
            cached_subgraph_string = self.cache_manager.get_data(
                meta_key=self.cache_key,
                dict_key=str(
                    (root_entity.uid, self.cache_manager.get_hash_value(str(options)))),
            )
            if cached_subgraph_string:
                try:
                    subgraph = Subgraph.model_validate(cached_subgraph_string)
                    logger.debug(
                        f"Loaded cached subgraph for entity {root_entity.uid}")
                    return root_entity, subgraph
                except Exception as e:
                    logger.error(
                        f"Could not load cached subgraph for entity {root_entity.uid}. Generating new subgraph...: {e}")

        # Use a BFS variant to get a subgraph
        # see https://en.wikipedia.org/wiki/Breadth-first_search for pseudo code
        # and more information.
        subgraph = self._bfs_build_subgraph(root_entity, options)
        subgraph = self._remove_duplicates(subgraph)

        # Add the subgraph to the cache
        self.cache_manager.add_data(
            meta_key=self.cache_key,
            dict_key=str(
                (root_entity.uid, self.cache_manager.get_hash_value(str(options)))),
            value=subgraph.model_dump()
        )

        return root_entity, subgraph

    def get_subgraphs_that_include_entity(self,
                                          root_entities: List[str | Knowledge],
                                          entity_to_include: Knowledge,
                                          use_caching: bool = False) -> Tuple[dict, dict]:
        """
        Builds subgraphs for all the given root entities based on the knowledge graph.
        It builds a dictionary with the root entity as key and the subgraph as value.
        However it only includes subgraphs if they contain the topic entity.

        Args:
            root_entities (List[str]): The root entity ids to build subgraphs from.
            entity_to_include (Knowledge): The topic entity to include in the subgraphs.
            use_caching (bool): Whether to use caching for the subgraphs.

        Returns:
            dict: A dictionary which maps entities to their corresponding
                subgraphs. The structure of the dictionary is as follows:
                {
                    "root_entity": Subgraph,
                    ...
                }
            dict: A dictionary which maps the root entity to the maximum amount of hops
                from the topic entity to the root entity. The structure is as follows:
                {
                    "root_entity": int,
                    ...
                }
        """
        # Next we build the subgraphs for the entities
        max_hop_amount_from_topic = {}
        restriction_subgraphs = {}
        for restriction_entity in root_entities:
            # if restriction_entity in self.restriction_subgraphs:
            #     continue

            if isinstance(restriction_entity, str):
                restriction_knowledge = self.knowledge_graph.get_entity_by_id(
                    restriction_entity)
            else:
                restriction_knowledge = restriction_entity

            paper_entity = self.knowledge_graph.get_paper_from_entity(
                restriction_knowledge)

            _, restriction_subgraph = self.get_subgraph(
                root_entity=paper_entity,
                options=SubgraphOptions(
                    go_against_direction=True,
                ),
                use_caching=use_caching
            )

            subgraph_paths = PathBuilder(restriction_subgraph).build_all_paths(
                current=restriction_knowledge,
                include_tails=True,
                include_against_direction=True
            )

            # Now we check if the topic entity is in the subgraph
            # and also calculate the amount of hops from the topic entity
            # to the root entity
            topic_in_subgraph, hop_amount_from_topic = self._check_topic_entity_and_calculate_hop_amount(
                subgraph_paths=subgraph_paths,
                entity_to_include=entity_to_include,
                restriction_knowledge=restriction_knowledge
            )

            if not topic_in_subgraph or hop_amount_from_topic < 0:
                continue
            max_hop_amount_from_topic[paper_entity] = hop_amount_from_topic
            logger.debug(
                f"Found subgraph for paper entity: {paper_entity}")
            restriction_subgraphs[paper_entity] = restriction_subgraph

        return restriction_subgraphs, max_hop_amount_from_topic

    def _check_topic_entity_and_calculate_hop_amount(self,
                                                     subgraph_paths: List[List[Triple]],
                                                     entity_to_include: Knowledge,
                                                     restriction_knowledge: Knowledge) -> Tuple[bool, int]:
        """
        Helper function to check if the topic entity is in the subgraph
        and calculate the amount of hops from the topic entity to the root entity.

        Args:
            subgraph_paths (List[List[Triple]]): The paths of the subgraph.
            entity_to_include (Knowledge): The topic entity to include in the subgraph.
            restriction_knowledge (Knowledge): The root entity of the subgraph.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating if the topic entity is in the subgraph
                and the amount of hops from the topic entity to the root entity.
        """
        topic_in_subgraph = False
        hop_amount_from_topic = -1
        for path in subgraph_paths:
            topic_index = -1
            restriction_index = -1
            for i, triple in enumerate(path):
                if (entity_to_include.uid in triple.entity_subject.uid or
                        entity_to_include.uid in triple.entity_object.uid):
                    topic_index = i
                if (restriction_knowledge.uid in triple.entity_subject.uid or
                        restriction_knowledge.uid in triple.entity_object.uid):
                    restriction_index = i
                if topic_index != -1 and restriction_index != -1:
                    break

            if topic_index != -1 and restriction_index != -1:
                topic_in_subgraph = True
                hops = abs(topic_index - restriction_index)
                if hop_amount_from_topic == -1 or hops < hop_amount_from_topic:
                    hop_amount_from_topic = hops
        return topic_in_subgraph, hop_amount_from_topic

    def _bfs_build_subgraph(self,
                            root_entity: Knowledge,
                            options: SubgraphOptions) -> Subgraph:
        """
        A BFS traversal of the knowledge graph.
        see https://en.wikipedia.org/wiki/Breadth-first_search for pseudo code
        and more information.

        Args:
            root_entity (Knowledge): The root entity to start the BFS from.
            options (SubgraphOptions): Configuration options for subgraph extraction.

        Returns:
            Subgraph: The constructed subgraph from the BFS traversal.
        """
        visited: Set[str] = set()
        subgraph: List[Triple] = []
        queue: deque = deque([(root_entity, 0)])

        self._prepare_progress_handler(queue)

        logger.debug("Building Subgraph with BFS")

        tail_relations = []
        while queue:
            current_entity, current_depth = queue.popleft()
            if self._should_stop(options.size_limit, subgraph):
                break
            if self._should_skip(
                    current_entity, options.hop_amount, current_depth, visited):
                self.progress_handler.update_task_by_string_id(
                    self.task, 1, False)
                continue

            visited.add(current_entity.uid)
            # Get all relations of the root entity
            if current_entity == root_entity or options.traverse_all_possible_edges:
                root_head_relations = self.knowledge_graph.get_relations_of_head_entity(
                    current_entity)
                relations = root_head_relations
                if options.go_against_direction:
                    root_tail_relations = self.knowledge_graph.get_relations_of_tail_entity(
                        current_entity)
                    tail_relations.extend(root_tail_relations)
                    relations = list(
                        set(root_head_relations + root_tail_relations))
            elif current_entity.uid in [relation.entity_subject.uid for relation in tail_relations]:
                tail_relations.extend(
                    self.knowledge_graph.get_relations_of_tail_entity(current_entity))
                relations = tail_relations
            else:
                relations = self.knowledge_graph.get_relations_of_head_entity(
                    current_entity)

            # Filter to only keep a specific amount of relations that include
            # the filter types (e.g. only keep 2 publications)
            relations = self._filter_relations_by_type(
                relations, options.filter_types, options.amount_of_filter_types_to_keep)

            subgraph.extend(relations)
            new_tail_entities = []
            for relation in relations:
                if relation.entity_subject.uid == current_entity.uid:
                    connected_entity = relation.entity_object
                else:
                    connected_entity = relation.entity_subject
                # Append the tail entity to the queue with an increased depth
                if connected_entity.uid not in visited:
                    new_tail_entities.append(
                        (connected_entity, current_depth + 1))

            # We do a random shuffle here to spread out the variations
            # of subgraphs that are generated if the size is limited
            if options.size_limit > 0:
                random.shuffle(new_tail_entities)

            queue.extend(new_tail_entities)
            self.progress_handler.update_task_length(self.task, len(queue))
            self.progress_handler.update_task_by_string_id(self.task, 1, False)
        self.progress_handler.finish_by_string_id(self.task)
        self.task = None
        return Subgraph(subgraph)

    def _get_names_for_relations(self, relations: List[Triple]):
        """
        Gets the names for the relations in the subgraph.
        This is done by looking up the entity in the knowledge graph
        and getting the name of the entity.

        Args:
            relations (List[Triple]): The relations to get the names for.
        """
        for relation in relations:
            relation.entity_subject.text = self.knowledge_graph.get_entity_by_id(
                relation.entity_subject.uid).text
            relation.entity_object.text = self.knowledge_graph.get_entity_by_id(
                relation.entity_object.uid).text

    def _filter_relations_by_type(self,
                                  relations: List[Triple],
                                  filter_types: List[str],
                                  amount_to_keep: int) -> List[Triple]:
        """
        Searches the relations for the filter types and keeps the amount_to_keep
        amount of relations that have the filter types. All other relations are
        kept and not affected by the filtering.

        Args:
            relations (List[Triple]): The relations to filter.
            filter_types (List[Tuple[str, str]]): The filter types to search for.
            amount_to_keep (int): The amount of relations to keep that have 
                the filter types.

        Returns:
            List[Triple]: The filtered relations.
        """
        if not filter_types:
            return relations
        relations_without_type = []
        filtered_relations_by_type = []
        for relation in relations:
            has_filter_type = False
            for filter_type in filter_types:
                if filter_type in relation.predicate:
                    filtered_relations_by_type.append(relation)
                    has_filter_type = True
            if not has_filter_type:
                relations_without_type.append(relation)
        if len(filtered_relations_by_type) > amount_to_keep:
            random.shuffle(filtered_relations_by_type)
            filtered_relations_by_type = filtered_relations_by_type[:amount_to_keep]

        filtered_relations_by_type.extend(relations_without_type)

        return filtered_relations_by_type

    def _prepare_progress_handler(self, queue: deque):
        """
        Prepares the progress handler for the subgraph building process.

        Args:
            queue (deque): The queue of entities to process.
        """
        self.task = self.progress_handler.add_task(
            string_id="subgraph_traversal",
            description="Building Subgraph",
            total=len(queue)
        )

    def _remove_duplicates(self, subgraph: Subgraph) -> Subgraph:
        """
        Removes duplicates from the subgraph.
        This is done by converting the subgraph to a set and back to a list.

        Args:
            subgraph (Subgraph): The subgraph to remove duplicates from.
        """
        subgraph = list(set(subgraph))
        return Subgraph(subgraph)

    def _select_root_entity(self, root_entity: Optional[Knowledge]) -> Knowledge:
        """
        Selects the root entity for the subgraph building process.

        Args:
            root_entity (Optional[Knowledge]): The root entity to select. If None, a random publication is chosen.

        Returns:
            Knowledge: The selected root entity.
        """
        if root_entity is None:
            return self.knowledge_graph.get_random_publication()
        return root_entity

    def _should_stop(self, size_limit: int, subgraph: Subgraph) -> bool:
        """
        Checks if the BFS should stop.

        Args:
            size_limit (int): The size limit for the subgraph.
            subgraph (Subgraph): The current subgraph.

        Returns:
            bool: True if the BFS should stop, False otherwise.
        """
        if size_limit == -1:
            return False
        return 0 < size_limit <= len(subgraph)

    def _should_skip(self,
                     current_entity: Knowledge,
                     hop_amount: int,
                     current_depth: int,
                     visited: Set[str]) -> bool:
        """
        Checks if the BFS should skip the current entity.

        Args:
            current_entity (Knowledge): The current entity to check.
            hop_amount (int): The hop amount for the BFS.
            current_depth (int): The current depth of the BFS.
            visited (Set[str]): The set of visited entities.

        Returns:
            bool: True if the BFS should skip the current entity, False otherwise.
        """
        # Check if the current depth is within the hop amount
        if 0 < hop_amount < current_depth:
            return True
        # Check if the current entity has been visited
        if current_entity.uid in visited:
            return True
        if not self.knowledge_graph.is_intermediate_id(current_entity.uid):
            return True
        return False
