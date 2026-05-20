from dataclasses import dataclass, field
from collections import deque
import random
from typing import List, Dict, Optional

from sqa_system.knowledge_base.knowledge_graph.storage.utils.path_builder import PathBuilder
from sqa_system.core.data.models import Triple, Knowledge, Subgraph
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GraphPathFilterOptions:
    """
    The options for the GraphPathFilter class.

    Args:
        word_blacklist (List[str]): A list of words that will be removed from the context
                the LLM is given to generate questions. The intention is to remove words that
                are not relevant from the context to not have the LLM generate questions based
                on them.
        size_limit (int): The maximum amount of relations that the generated
    """
    word_blacklist: Optional[List[str]] = field(default_factory=list)
    size_limit: Optional[int] = None


class GraphPathFilter:
    """
    A class responsible for filtering paths in a subgraph.
    """

    def filter_paths_by_function(self,
                                 paths: List[List[Triple]],
                                 predicate_to_filter_by: str,
                                 filtering_function: callable = None) -> List[List[Triple]]:
        """
        Filters paths based on a specified predicate and filtering function.

        Args:
            paths (List[List[Triple]]): List of paths to filter, where each path is 
                a list of Triples
            predicate_to_filter_by (str): The predicate to look for in the paths
            filtering_function (callable): A function that takes the predicate object  
                value and returns bool

        Returns:
            List[List[Triple]]: Filtered list of paths that match the criteria
        """
        filtered_paths = []
        for path in paths:
            predicate_found = False
            for triple in path:
                if predicate_to_filter_by in triple.predicate:
                    predicate_found = True
                    predicate_object_value = triple.entity_object.text
                    if filtering_function and filtering_function(predicate_object_value):
                        filtered_paths.append(path)
                    break

            if not predicate_found:
                continue

        return filtered_paths

    def filter_paths_by_type_and_name(self,
                                      root: Knowledge,
                                      subgraph: Subgraph,
                                      filter_type: str,
                                      keep_names: List[str] = None) -> Subgraph:
        """
        Filters out paths from the subgraph that contain an entity with the given type.
        It keeps the paths that contain the given names.

        Args:
            root (Knowledge): The root entity of the subgraph.
            subgraph (List[Relation]): The subgraph to filter.
            filter_type (str): The type to filter out.
            keep_names (List[str]): The list of types with names to keep

        Returns:
            Subgraph: The filtered subgraph.
        """
        # Build all paths from the subgraph
        path_builder = PathBuilder(subgraph=subgraph)
        subgraph_paths = path_builder.build_all_paths(root, include_tails=True)

        filtered_paths = []
        for path in subgraph_paths:
            # Collect entities in the path whose type matches filter_type
            matched_entities = []
            for triple in path:
                for entity in (triple.entity_subject, triple.entity_object):
                    if filter_type in entity.knowledge_types:
                        matched_entities.append(entity)

            # If no entity in the path matches filter_type, we keep the path.
            if not matched_entities:
                filtered_paths.append(path)
            else:
                if keep_names:
                    # Keep the path if at least one matched entity has a name in keep_names.
                    if any(entity.text in keep_names for entity in matched_entities):
                        filtered_paths.append(path)

        filtered_subgraph = self.get_subgraph_from_paths(filtered_paths)
        return filtered_subgraph

    def filter_paths_by_type_substring(self,
                                       root: Knowledge,
                                       subgraph: Subgraph,
                                       filter_type: str) -> Subgraph:
        """
        Filters out paths from the subgraph that contain an entity with the given type.

        Args:
            root (Knowledge): The root entity of the subgraph.
            subgraph (List[Relation]): The subgraph to filter.
            filter_type (str): The type to filter out.

        Returns:
            Subgraph: The filtered subgraph.
        """
        path_builder = PathBuilder(subgraph=subgraph)
        subgraph_paths = path_builder.build_all_paths(root, include_tails=True)

        filtered_paths = []
        for path in subgraph_paths:
            found = False
            for triple in path:
                for entity in (triple.entity_subject, triple.entity_object):
                    for entity_type in entity.knowledge_types:
                        if filter_type.lower() in entity_type.lower():
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                filtered_paths.append(path)

        filtered_subgraph = self.get_subgraph_from_paths(filtered_paths)
        return filtered_subgraph

    def filter_paths_by_entity(self,
                               root: Knowledge,
                               subgraph: Subgraph,
                               filter_list: List[Knowledge],
                               keep_paths: bool = False) -> Subgraph:
        """
        Filters out paths from the subgraph that contain an entity that is in the filter list.

        Args:
            root (Knowledge): The root entity of the subgraph.
            subgraph (List[Relation]): The subgraph to filter.
            filter_list (List[Knowledge]): The list of entities to filter out.
            keep_paths (bool): Whether to keep the paths that contain the filter entities or remove
                them. If True, the paths that contain the filter entities are kept and all other
                paths are removed.

        Returns:
            Subgraph: The filtered subgraph.
        """
        if not filter_list:
            return subgraph

        # Build all paths from the subgraph
        path_builder = PathBuilder(subgraph=subgraph)
        subgraph_paths = path_builder.build_all_paths(root, include_tails=True)

        # Filter paths containing entities from filter_list
        filtered_paths = []
        for path in subgraph_paths:
            path_contains_filter = False
            for triple in path:
                if (triple.entity_subject in filter_list or
                        triple.entity_object in filter_list):
                    path_contains_filter = True
                    break
            if keep_paths and path_contains_filter:
                filtered_paths.append(path)
            elif not keep_paths and not path_contains_filter:
                filtered_paths.append(path)

        # Convert filtered paths back to subgraph
        filtered_subgraph = self.get_subgraph_from_paths(filtered_paths)

        return filtered_subgraph

    def filter_paths_by_predicate(self,
                                  root: Knowledge,
                                  subgraph: List[Triple],
                                  filter_list: List[str],
                                  keep_paths: bool = False) -> List[Triple]:
        """
        Filters out paths from the subgraph that contain a predicate that is in the filter list.

        Args:
            root (Knowledge): The root entity of the subgraph.
            subgraph (List[Relation]): The subgraph to filter.
            filter_list (List[str]): The list of predicates to filter out.
            keep_paths (bool): Whether to keep the paths that contain the filter predicates or remove
                them. If True, the paths that contain the filter predicates are kept and all other
                paths are removed.

        Returns:
            List[Triple]: The filtered subgraph as a list of Triple objects.
        """
        if not filter_list:
            return subgraph
        path_builder = PathBuilder(subgraph=subgraph)
        subgraph_paths = path_builder.build_all_paths(root, include_tails=True)
        filtered_paths = []
        for path in subgraph_paths:
            path_contains_filter = False
            for triple in path:
                if triple.predicate in filter_list:
                    path_contains_filter = True
                    break
            if keep_paths and path_contains_filter:
                filtered_paths.append(path)
            elif not keep_paths and not path_contains_filter:
                filtered_paths.append(path)
        filtered_subgraph = self.get_subgraph_from_paths(filtered_paths)
        return filtered_subgraph

    def get_subgraph_from_paths(self, paths: List[List[Triple]]) -> Subgraph:
        """
        Converts a list of paths into a subgraph by removing duplicate relations.
        Each path is a list of Triple objects, and the function ensures that
        each relation in the subgraph is unique based on the combination of
        entity_subject, predicate, and entity_object.

        Args:
            paths (List[List[Triple]]): A list of paths, where each path is a list of Triple objects.

        Returns:
            Subgraph: The resulting subgraph containing unique relations.        
        """
        unique_relations = {}

        for path in paths:
            for relation in path:
                if relation.entity_subject is None or relation.entity_object is None:
                    continue
                relation_key = (relation.entity_subject.uid,
                                relation.predicate, relation.entity_object.uid)

                if relation_key not in unique_relations:
                    unique_relations[relation_key] = relation

        return list(Subgraph(unique_relations.values()))

    def limit_paths(self,
                    root_entity: Knowledge,
                    filtered_paths: List[List[Triple]],
                    size_limit: int) -> List[List[Triple]]:
        """
        The LLM can't receive a subgraph that is too large.
        Therefore we have to reduce the amount of paths to a certain
        limit. However, we need to carefully think about which
        paths to keep and which to remove as this information is what
        the LLM uses to generate the question and answer.
        We observed that randomly selecting paths leads to a bad performance.
        We assume that the reason is that for a node not all its edges are
        in the final graph. Therefore, we now apply a certain algorithm that
        tries to include the edges of each node added to the graph.

        Args:
            root_entity (Knowledge): The root entity of the subgraph.
            filtered_paths (List[List[Triple]]): The list of paths to filter.
            size_limit (int): The maximum number of paths to keep.

        Returns:
            List[List[Triple]]: The filtered paths, limited to the specified size.
        """
        # We select a path that includes the root entity
        selected_path = self._select_path_with_root(
            root_entity, filtered_paths)
        final_paths = [selected_path]

        # Now we build a map of nodes to their connected edges
        node_to_paths: Dict[str, List[List[Triple]]] = self._build_map_of_nodes_to_edges(
            filtered_paths)

        # Now we build a map of head entities for each tail entity
        head_entities: Dict[str, set[Knowledge]] = self._build_map_of_heads_to_tails(
            filtered_paths)

        # Now we process each relation in the selected path
        # We need to make sure that we start from the right side
        visited_heads = set()
        # We disable the too-many-nested-blocks warning here
        # because it would make the code less readable if we
        # tried to split up the following code.
        # pylint: disable=too-many-nested-blocks
        for relation in reversed(selected_path):
            heads_queue = deque()
            if relation.entity_subject and relation.entity_subject.uid:
                heads_queue.extend(head_entities.get(
                    relation.entity_subject.uid, []))

            while len(heads_queue) > 0:
                head = heads_queue.pop()
                if head in visited_heads:
                    continue
                visited_heads.add(head)
                # We add the paths of the head entity to the final paths
                for path in node_to_paths.get(head.uid, []):
                    if path not in final_paths:
                        final_paths.append(path)
                        if len(final_paths) >= size_limit:
                            break
                        # We add the head entities of the path to the queue
                        for rel in path:
                            if (rel.entity_subject and
                                rel.entity_subject.uid and
                                    rel.entity_subject not in visited_heads):
                                heads_queue.extend(head_entities.get(
                                    rel.entity_subject.uid, []))
                if len(final_paths) >= size_limit:
                    break
            if len(final_paths) >= size_limit:
                break
        return final_paths

    def _select_path_with_root(self,
                               root_entity: Knowledge,
                               filtered_paths: List[List[Triple]]) -> List[Triple]:
        """
        Selects a path that contains the root entity.

        Args:
            root_entity (Knowledge): The root entity of the subgraph.
            filtered_paths (List[List[Triple]]): The list of paths to select from.

        Returns:
            List[Triple]: A path that contains the root entity.
        """
        selected_path = None
        copy_of_paths = filtered_paths.copy()
        random.shuffle(copy_of_paths)
        for path in copy_of_paths:
            if root_entity in [relation.entity_subject for relation in path]:
                selected_path = path
                break
        if selected_path is None:
            selected_path = random.choice(filtered_paths)
        return selected_path

    def _build_map_of_nodes_to_edges(
            self,
            filtered_paths: List[List[Triple]]) -> Dict[str, List[List[Triple]]]:
        """
        Builds a mapping where the key is the node and the value is a list of paths
        that contain the node.

        Args:
            filtered_paths (List[List[Triple]]): A list of paths.

        Returns:
            Dict[str, List[List[Triple]]]: A dictionary mapping nodes to paths.
        """
        node_to_paths: Dict[str, List[List[Triple]]] = {}
        for path in filtered_paths:
            for relation in path:
                if (relation.entity_subject is not None and
                        relation.entity_subject.uid is not None):
                    if relation.entity_subject.uid not in node_to_paths:
                        node_to_paths[relation.entity_subject.uid] = []
                    node_to_paths[relation.entity_subject.uid].append(path)

                if (relation.entity_object is not None and
                        relation.entity_object.uid is not None):
                    if relation.entity_object.uid not in node_to_paths:
                        node_to_paths[relation.entity_object.uid] = []
                    node_to_paths[relation.entity_object.uid].append(path)
        return node_to_paths

    def _build_map_of_heads_to_tails(
            self,
            filtered_paths: List[List[Triple]]) -> Dict[str, set[Knowledge]]:
        """
        Builds a mapping from head entity UIDs to their corresponding Knowledge 
        objects.

        Args:
            filtered_paths (List[List[Triple]]): A list of filtered paths, 
                each path is a list of Triple objects.

        Returns:
            Dict[str, set[Knowledge]]: A dictionary where the keys are head entity UIDs
        """
        head_entities: Dict[str, set[Knowledge]] = {}
        for path in filtered_paths:
            for relation in path:
                if (relation.entity_subject is not None and
                    relation.entity_subject.uid is not None and
                        relation.entity_subject.uid not in head_entities):
                    head_entities[relation.entity_subject.uid] = set()
                if (relation.entity_subject is not None and
                    relation.entity_subject.uid is not None and
                        relation.entity_object is not None):
                    head_entities[relation.entity_subject.uid].add(
                        relation.entity_object)
        return head_entities

    def _remove_blacklisted_paths(self,
                                  subgraph_paths: List[List[Triple]],
                                  word_blacklist: List[str]) -> List[List[Triple]]:
        """
        Removes paths that contain any blacklisted words in any triple contained.

        Args:
            subgraph_paths (List[List[Triple]]): The list of subgraph paths to filter.
            word_blacklist (List[str]): A list of words to blacklist.

        Returns:
            List[List[Triple]]: The filtered list of subgraph paths.
        """
        if not word_blacklist:
            return subgraph_paths
        paths_to_remove = []
        for path in subgraph_paths:
            for relation in path:
                if (relation.entity_subject is None or
                        relation.entity_object is None or
                        relation.predicate is None):
                    continue
                for word in word_blacklist:
                    word_to_check = word.lower()
                    if (relation.entity_subject.text is not None and
                            word_to_check in relation.entity_subject.text.lower()):
                        paths_to_remove.append(path)
                        break
                    if (relation.entity_object.text is not None and
                            word_to_check in relation.entity_object.text.lower()):
                        paths_to_remove.append(path)
                        break
                    if (relation.predicate is not None and
                            word_to_check in relation.predicate.lower()):
                        paths_to_remove.append(path)
                        break
        subgraph_paths = [
            path for path in subgraph_paths if path not in paths_to_remove]
        return subgraph_paths
