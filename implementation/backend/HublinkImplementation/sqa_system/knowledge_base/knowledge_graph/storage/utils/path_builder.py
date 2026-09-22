from typing import List, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from sqa_system.core.data.models import (
    Triple,
    Knowledge,
    Subgraph
)
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TraversalOptions:
    """
    Represents the options for the traversal.
    
    Args:
        include_tails (bool): Whether the path-building process should consider
            relations where the current node is the object of the relation.
        include_against_direction (bool): Allows to explore paths that "turn back"
            after traversing a tail relation. It has only an effect if "include_tails" 
            is set to true.
    """
    include_tails: bool
    include_against_direction: bool = False


@dataclass
class TraversalState:
    """
    Represents the current state of the traversal.
    
    Args:
        node (Knowledge): The current node in the traversal.
        path (List[Triple]): The path taken to reach the current node.
        direction (str): The direction of the traversal ("right" or "left").
    """
    node: Knowledge
    path: List[Triple] = field(default_factory=list)
    direction: str = "right"


class PathBuilder:
    """
    This class is responsible for building paths from a given sub-
    graph of triples.

    Args:
        subgraph (List[Triple]): List of relations to build paths from.
    """

    def __init__(self, subgraph: Subgraph):
        self.subgraph = subgraph
        self.visited_nodes = set()
        self.progress_handler = ProgressHandler()

    def build_paths_from_root_entity(self,
                                     root_entities: Set[Knowledge]) -> List[List[Triple]]:
        """
        Builds all possible paths from the instance's subgraph, with each of the
        root_entities serving as a starting point for path discovery.

        Args:
            root_entities (Set[Knowledge]): The set of entities to use as starting
                points for path building.

        Returns:
            List[List[Triple]]: A list of paths, where each path is a list of triples.
        """
        logger.debug("Building Entity Paths")
        task_id = self.progress_handler.add_task(
            description="Finding all relevant paths in the graph..",
            string_id="finding_paths",
            total=len(root_entities)
        )
        all_paths: List[List[Triple]] = []
        path_builder = PathBuilder(self.subgraph)
        for index, entity in enumerate(root_entities):
            logger.debug(
                f"Processing Entity: {index}|{len(root_entities)}")
            all_paths.extend(path_builder.build_all_paths(
                current=entity,
                include_tails=True
            ))
            self.progress_handler.update_task_by_string_id(task_id, 1)
        self.progress_handler.finish_by_string_id(task_id)
        return all_paths

    def build_all_paths(self,
                        current: Knowledge,
                        include_tails: bool,
                        include_against_direction: bool = False) -> List[List[Triple]]:
        """
        This function builds all possible paths from the given subgraph.

        Args:
            current: The start node in the Subgraph from which the path 
                should be built.
            include_tails: Whether to include tail relations
            include_against_direction: Whether to include relations against
                the direction during the traversal. Only works if include_tails
                is set to true.
        Returns:
            List[List[Triple]]: A list of paths, where each path is a list of triples.
        """
        options = TraversalOptions(
            include_tails=include_tails,
            include_against_direction=include_against_direction
        )
        return self._build_all_paths_bfs(
            start_node=current,
            options=options
        )

    def _build_all_paths_bfs(
            self,
            start_node: Knowledge,
            options: TraversalOptions) -> List[List[Triple]]:
        """
        Builds all possible paths from the subgraph using Breadth-First Search (BFS),
        starting from the given node.
        Code adapted from: https://en.wikipedia.org/wiki/Breadth-first_search

        Args:
            start_node (Knowledge): The node in the subgraph from which to start building paths.
            options (TraversalOptions): Configuration for the traversal, such as
                whether to include tail relations or traverse against the relation direction.
        Returns:
            List[List[Triple]]: A list of paths, where each path is a list of triples.
        """
        self.visited_nodes = set()
        queue = deque([
            TraversalState(
                node=start_node,
                path=[],
                direction="right"
            )])

        all_paths: List[List[Triple]] = []

        while queue:
            # Dequeue the first element in the queue
            current_state = queue.popleft()

            if current_state.node in self.visited_nodes:
                all_paths.append(current_state.path)
                continue

            # Add current node to visited set
            self.visited_nodes.add(current_state.node)

            # Retrieve the current relations
            tail_relations, current_relations = self._get_current_relations(
                current_state.node,
                options.include_tails,
                current_state.direction
            )

            if not current_relations:
                # If there are no further relations to explore from this node,
                # add the current path to all_paths
                all_paths.append(current_state.path)
                continue

            for relation in current_relations:

                result = self._determine_next_nodes(
                    relation, tail_relations, current_state, options
                )
                next_node, next_direction, new_path, new_state = result
                if new_state:
                    queue.append(new_state)

                # Continue only if the next node exists and hasn't been visited
                if next_node:
                    # Enqueue the new state with the next node, updated path, and new direction
                    queue.append(TraversalState(
                        node=next_node,
                        path=new_path,
                        direction=next_direction
                    ))
        return all_paths

    def _determine_next_nodes(self,
                              relation: Triple,
                              tail_relations: List[Triple],
                              current_state: TraversalState,
                              options: TraversalOptions):
        """
        Helper function to determine the next node(s) and direction(s)
        based on the current relation and traversal options.

        Args:
            relation: The current triple being traversed.
            tail_relations: A list of triples where the current node is the object.
            current_state: The current state of the traversal (node, path, direction).
            options: The traversal options.

        Returns:
            Tuple[Knowledge | None, str, List[Triple], TraversalState | None]:
                A tuple containing:
                - The primary next node to visit (or None if not applicable).
                - The direction for the primary next step ("left" or "right").
                - The new path including the current relation.
                - An optional additional TraversalState for branching (e.g., when
                  `include_against_direction` is True and a tail relation is encountered),
                  or None if no branching occurs.
        """
        # Create a new path by appending the current relation
        new_path = current_state.path + [relation]
        new_state = None
        # Determine the next node and the direction based on the current relation
        if options.include_tails:
            if relation in tail_relations:
                # If the relation is a tail relation, traverse to the subject
                next_node = relation.entity_subject
                next_direction = "left"

                # If include_against_direction is True, also consider
                # traversing against direction
                if options.include_against_direction:
                    # Traverse back to the object with the opposite direction
                    back_node = relation.entity_object
                    if back_node and back_node not in self.visited_nodes:
                        # Create a new path for the back traversal
                        back_path = new_path.copy()
                        # Enqueue the back traversal
                        new_state = TraversalState(
                            node=back_node,
                            path=back_path,
                            direction="right"
                        )
            else:
                # If the relation is not a tail relation, traverse to the object
                next_node = relation.entity_object
                next_direction = "right"
        else:
            if current_state.direction == "right":
                next_node = relation.entity_object
                next_direction = "right"
            else:
                next_node = relation.entity_subject
                next_direction = "left"
        return next_node, next_direction, new_path, new_state

    def _get_current_relations(
            self,
            current: Knowledge,
            include_tails: bool = False,
            direction: str = "right") -> Tuple[List[Triple], List[Triple]]:
        """
        Retrieves relations connected to the current node in the subgraph,
        considering traversal options and visited nodes.
        
        Args:
            current: The current node in the subgraph.
            include_tails: Whether to include relations where the current node
                is the object (tail relations).
            direction: The current direction of traversal ("right" for subject to object,
                "left" for object to subject). This influences which relations are
                considered if `include_tails` is False.
        Returns:
            Tuple[List[Triple], List[Triple]]: A tuple containing two lists:
                - tail_relations (List[Triple]): A list of relations where `current`
                  is the object and the subject has not been visited. Empty if
                  `include_tails` is False.
                - current_relations (List[Triple]): A list of relevant relations
                  (both head and tail if `include_tails` is True, otherwise filtered
                  by `direction`) connected to `current` where the other entity in
                  the relation has not been visited.
        """
        tail_relations = []
        if include_tails:
            current_relations = [
                relation for relation in self.subgraph
                if (relation.entity_subject == current and
                    relation.entity_object not in self.visited_nodes)]
            tail_relations = [
                relation for relation in self.subgraph
                if (relation.entity_object == current and
                    relation.entity_subject not in self.visited_nodes)]
            current_relations += tail_relations
        elif direction == "right":
            current_relations = [
                relation for relation in self.subgraph
                if (relation.entity_subject == current and
                    relation.entity_object not in self.visited_nodes)]
        else:
            current_relations = [
                relation for relation in self.subgraph
                if (relation.entity_object == current and
                    relation.entity_subject not in self.visited_nodes)]
        return tail_relations, current_relations
