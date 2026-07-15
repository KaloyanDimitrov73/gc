from collections import deque
from typing import List, Tuple

from core.data.models.knowledge import Knowledge
from core.data.models.triple import Triple
from core.progress.progress_handler import ProgressHandler
from knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from core.logging.logging import get_logger

from ..models import EntityWithDirection, Hub, IsHubOptions

logger = get_logger(__name__)


class HubFinder:
    """
    Finds hub entities in a knowledge graph for use in hub-based retrieval.

    The HubFinder traverses the graph from given root entities, identifying hubs according to the provided options.

    Args:
        graph (KnowledgeGraph): The knowledge graph to search.
        is_hub_options (IsHubOptions): Options for classifying entities as hubs.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 is_hub_options: IsHubOptions):
        self.graph = graph
        self.progress_handler = ProgressHandler()
        self.visited_with_direction = set()
        self.is_hub_options = is_hub_options

    def find_hub_root_entities(self,
                  root_entities: List[EntityWithDirection],
                  reset_memory: bool = False,
                  need_same_hop_amount: bool = True
                  ) -> Tuple[List[EntityWithDirection], List[EntityWithDirection]]:
        """
        Starting from the root entities, this method searches their paths
        until for each path either the end is reached or a hub entity is found. 

        Args:
            root_entities (List[EntitywithDirection]): The entities to start the search from.
            reset_memory (bool): Whether to reset the memory of the finder.
            need_same_hop_amount (bool): If set to true, the hubs that are returned all 
                have the same amount of hops from the root entities.
        Returns:
            Tuple[List[EntitywithDirection], List[EntitywithDirection]]: 
                The first list contains the hub entities that were found.
                The second list contains the entities that are 
                candidates for the next traversal.
        """
        if reset_memory:
            self.visited_with_direction.clear()

        queue: deque = deque()
        queue.extend(root_entities)
        total = len(queue)

        hub_task = self.progress_handler.add_task(
            description="Searching for Hubs..",
            total=len(queue),
            string_id="hub_search"
        )

        # Search for hubs
        found_hub_entities = []
        next_traversal_candidates = []
        logger.info("Starting to search for hubs in the graph")
        while queue:
            entity_with_direction: EntityWithDirection = queue.popleft()

            # Determine whether the entity should be skipped
            visited_key = (entity_with_direction.entity.uid,
                           entity_with_direction.left)
            if self._should_skip_entity(entity_with_direction,
                                        self.visited_with_direction):
                self.progress_handler.update_task_by_string_id(hub_task, advance=1)
                continue
            self.visited_with_direction.add(visited_key)

            # Check if the current entity is a hub
            is_hub = Hub.is_hub_entity(
                entity_with_direction.entity,
                self.graph,
                self.is_hub_options)

            if is_hub and entity_with_direction not in found_hub_entities:
                found_hub_entities.append(entity_with_direction)

            if is_hub and not entity_with_direction.left:
                # If we have a hub entity we do not need to consider
                # the relations of the entity to the right because they
                # are part of the hub itself.
                self.progress_handler.update_task_by_string_id(hub_task, advance=1)
                continue

            # Add relations of the entity to the queue considering the direction
            next_entities = self._get_next_relations(entity_with_direction)

            # We can either extend the Queue here or add the entities to the
            # next_traversal_candidates.
            # The difference is, that we either say that the Hub Levels are
            # also based on the Hop amount
            # or not.
            if need_same_hop_amount:
                next_traversal_candidates.extend(next_entities)
            else:
                queue.extend(next_entities)
                total += len(next_entities)
                self.progress_handler.update_task_length(hub_task, total)


            self.progress_handler.update_task_by_string_id(hub_task, advance=1)

        self.progress_handler.finish_by_string_id(hub_task)
        return found_hub_entities, next_traversal_candidates

    def _should_skip_entity(self,
                            entity_with_direction: EntityWithDirection,
                            visited_with_direction: set) -> bool:
        """
        Determines if the current entity should be skipped.
        This is done by using the uid of the entity but 
        also the direction in which the entity is traversed.
        
        Args:
            entity_with_direction (EntityWithDirection): The entity to check.
            visited_with_direction (set): The set of visited entities.
            
        Returns:
            bool: True if the entity should be skipped, False otherwise.
        """
        current_entity = entity_with_direction.entity
        is_left = entity_with_direction.left
        visited_key = (current_entity.uid, is_left)
        if visited_key in visited_with_direction:
            return True
        return False

    def _get_next_relations(
            self,
            entity_with_direction: EntityWithDirection) -> List[Tuple[Knowledge, Triple]]:
        """
        Get the next relations of the current entity.
        Also consider the direction in which the entity is traversed.
        
        Args:
            entity_with_direction (EntityWithDirection): The current entity.
            
        Returns:
            List[Tuple[Knowledge, Triple]]: The next relations of the current entity.
        """
        current_entity = entity_with_direction.entity
        is_left = entity_with_direction.left
        current_path = entity_with_direction.path_from_topic
        if is_left:
            relations = self.graph.get_relations_of_tail_entity(
                current_entity)
            return [
                EntityWithDirection(
                    entity=relation.entity_subject,
                    left=is_left,
                    path_from_topic=current_path + [relation]
                )
                for relation in relations
            ]

        relations = self.graph.get_relations_of_head_entity(
            current_entity)
        return [
            EntityWithDirection(
                entity=relation.entity_object,
                left=is_left,
                path_from_topic=current_path + [relation]
            )
            for relation in relations
        ]
