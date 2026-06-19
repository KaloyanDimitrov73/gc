import os
from typing import List
import re
import pandas as pd
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.data.models import Triple, Knowledge, Subgraph, QAPair
from sqa_system.core.config.models import KnowledgeGraphConfig
from sqa_system.knowledge_base.knowledge_graph.storage.knowledge_graph_manager import KnowledgeGraphManager
from sqa_system.core.data.data_loader.implementations.csv_qa_loader import CSVQALoader
from sqa_system.core.data.models.dataset.implementations.qa_dataset import QADataset
from sqa_system.knowledge_base.knowledge_graph.storage.utils import (
    SubgraphBuilder,
    SubgraphOptions
)

from ..strategies import ClusterBasedQuestionGenerator, ClusterGeneratorOptions
from .utils.qa_candidate_extractor import QACandidateExtractor
from .utils.qa_pair_updater import QAPairUpdater


class QADatasetToGraphConverter:
    """
    When the data from the ORKG is deleted and has to be uploaded again, 
    the QA dataset(s) for the graph are invalidated as the IDs of the 
    triples change.

    This issue also appears when moving or changing information in the graph.

    This class is responsible for converting the ids in the triples of a qa
    dataset to the current ids in the graph.

    Args:
        kg_config (KnowledgeGraphConfig): The configuration for the knowledge graph
            to convert to.
        string_replacements (dict[str, str], optional): A dictionary of
                string replacements to be applied to the candidates before
                matching. Has to be in lower case. They key is the string to
                be replaced and the value is the string to replace it with.
        update_in_place (bool, optional): Whether to update the QA dataset in place
            meaning that the CSV file will be overwritten. Defaults to False.
    """

    def __init__(self,
                 kg_config: KnowledgeGraphConfig,
                 string_replacements: dict[str, str] = None,
                 update_in_place: bool = False):
        self.kg_config = kg_config
        self.graph = KnowledgeGraphManager().get_item(config=kg_config)
        self.string_replacements = string_replacements if string_replacements else {}
        self.update_in_place = update_in_place

    def validate_qa_dataset(self, qa_dataset_path: str) -> bool:
        """
        This method validates whether the given QA dataset is valid meaning that all the 
        golden triples that are in the dataset are also in the knowledge graph.

        Args:
            qa_dataset_path (str): The path to the QA dataset to validate.

        Returns:
            bool: True if the QA dataset is valid, False otherwise.
        """
        if not os.path.exists(qa_dataset_path) or not os.path.isfile(qa_dataset_path):
            return False

        try:
            qa_pairs = CSVQALoader().load("", qa_dataset_path).get_all_entries()
            if not qa_pairs:
                print(f"No QA pairs found in {qa_dataset_path}")
                return False

            for qa_pair in qa_pairs:
                if not qa_pair.golden_triples:
                    print(f"QA pair {qa_pair.uid} has no golden triples.")
                    continue

                for triple in qa_pair.golden_triples:
                    # Pattern to extract ids from the ORKG triples
                    pattern = r'[R]\d+'
                    first_match = re.search(pattern, triple)
                    if not first_match:
                        continue
                    triple_id = first_match.group(0)
                    relations = self.graph.get_relations_of_head_entity(self.graph.get_entity_by_id(triple_id))
                    is_valid = False
                    for relation in relations:
                        if str(relation) == triple:
                            is_valid = True
                            break
                    if not is_valid:
                        return False

            print(f"QA dataset {qa_dataset_path} is valid.")
            return True
        except Exception as e:
            print(
                f"Error loading QA dataset from file. Maybe not a QA dataset? {e}")
            return False

    def run_conversion(self, qa_dataset_path: str, research_field_id: str):
        """
        This method runs the conversion of the QA dataset to the current
        knowledge graph. 

        Args:
            qa_dataset_path (str): The path to the QA dataset to convert.
            research_field_id (str): The ID of the research field entity from which
                the publication should be mapped to their nodes.
        """
        self._internal_run_conversion(
            qa_dataset_path=qa_dataset_path,
            doi_to_entity_mapping=self.get_doi_to_entity_mapping(research_field_id=research_field_id))

    def _internal_run_conversion(self, qa_dataset_path: str, doi_to_entity_mapping: dict[str, Knowledge]):
        if not os.path.exists(qa_dataset_path):
            print(f"QA dataset path does not exist: {qa_dataset_path}")
            return
        if not os.path.isfile(qa_dataset_path):
            print(f"QA dataset path is not a file: {qa_dataset_path}")
            return

        try:
            # Load the qa pairs
            qa_pairs = CSVQALoader().load("", qa_dataset_path).get_all_entries()
            print(f"Starting repairing {qa_dataset_path}...")

            updated_qa_pairs = []
            for index, qa_pair in enumerate(qa_pairs):
                print(f"Updating QA Dataset ({index}/ {len(qa_pairs)})")
                updated_qa_pair = self._update_qa_pair(
                    qa_pair=qa_pair,
                    doi_to_entity_mapping=doi_to_entity_mapping
                )
                if not updated_qa_pair:
                    raise ValueError(
                        f"Could not update QA Pair: {qa_pair.uid}.")
                updated_qa_pairs.append(updated_qa_pair)
            print(f"Finished repairing {qa_dataset_path}...")

            self._save_updated_qa_dataset(
                qa_dataset_path=qa_dataset_path,
                updated_qa_pairs=updated_qa_pairs
            )

        except Exception as e:
            print(
                f"Error loading QA datset from file. Maybe not a QA dataset? {e}")
            return

    def _update_qa_pair(self,
                        qa_pair: QAPair,
                        doi_to_entity_mapping: dict[str, Knowledge]) -> QAPair | None:
        """
        This method updates the given QA Pair by replacing the IDs of the triples
        and entities with the current IDs in the knowledge graph. 

        Args:
            qa_pair (QAPair): The QA Pair to update.
            doi_to_entity_mapping (dict[str, Knowledge]): A mapping of DOIs to
                their corresponding entities in the knowledge graph.

        Returns:
            QAPair | None: The updated QA Pair or None if the update was not
                successful.
        """
        updated_qa_pair: QAPair = qa_pair.model_copy(deep=True)

        # Get the sources of the QA Pair
        source_ids = qa_pair.source_ids
        if not source_ids or len(source_ids) == 0:
            print(
                f"Could not repair {qa_pair.uid} because of missing source ids")
            return None

        # Get the subgraph of the sources
        qa_pair_subgraph = self._get_subgraph_from_dois(
            dois=source_ids,
            doi_to_entity_mapping=doi_to_entity_mapping
        )
        if not qa_pair_subgraph:
            print(
                f"Could not repair {qa_pair.uid} because one of the sources was not found")
            return None

        # Get the updated triples
        new_topic_entity_candidates, new_golden_triple_candidates = self._get_updated_ground_truth(
            old_qa_pair=qa_pair,
            qa_pair_subgraph=qa_pair_subgraph
        )

        if not new_topic_entity_candidates or len(new_topic_entity_candidates) == 0:
            raise ValueError(
                f"Could not repair {qa_pair.question} because of missing topic entity candidates")

        if not new_golden_triple_candidates or len(new_golden_triple_candidates) == 0:
            raise ValueError(
                f"Could not repair {qa_pair.question} because of missing golden triple candidates")

        updater = QAPairUpdater(graph=self.graph)
        # Update the qa pair
        new_topic_entity = updater.update_topic_entity(
            old_qa_pair=qa_pair,
            updated_qa_pair=updated_qa_pair,
            new_topic_entity_candidates=new_topic_entity_candidates
        )
        updater.update_golden_tripels(
            old_qa_pair=qa_pair,
            updated_qa_pair=updated_qa_pair,
            new_golden_triple_candidates=new_golden_triple_candidates,
            string_replacements=self.string_replacements
        )
        updater.update_hop_amount(
            updated_qa_pair=updated_qa_pair,
            topic_entity=new_topic_entity,
            qa_pair_subgraph=qa_pair_subgraph,
            updated_golden_triples=updated_qa_pair.golden_triples
        )
        return updated_qa_pair

    def _get_updated_ground_truth(self,
                                  old_qa_pair: QAPair,
                                  qa_pair_subgraph: Subgraph) -> tuple[set[Knowledge], dict[str, set]]:
        """
        For the given QA Pair and its subgraph, this method extracts the updated 
        ground truth information from the subgraph and returns it.

        Args:
            old_qa_pair (QAPair): The original QA Pair.
            qa_pair_subgraph (Subgraph): The subgraph of the QA Pair.

        Returns:
            tuple[set[Knowledge], dict[str, set]]: A tuple containing a set of
                new topic entity candidates and a dictionary of new golden
                triple candidates.
        """
        new_topic_entity_candidates: set[Knowledge] = set()
        new_golden_triple_candidates: dict[str, set] = {}
        golden_triples = old_qa_pair.golden_triples.copy()
        visited_triples = set()
        candidate_extractor = QACandidateExtractor(
            predicate_mappings=self.string_replacements
        )
        for triple in qa_pair_subgraph:
            if triple in visited_triples:
                continue
            visited_triples.add(triple)

            # Check topic entity
            topic_entity_candidate = candidate_extractor.get_topic_candidate_from_triple(
                triple=triple,
                topic_entity_value=old_qa_pair.topic_entity_value
            )
            if topic_entity_candidate:
                new_topic_entity_candidates.add(topic_entity_candidate)

            # Check golden triples
            candidate_extractor.update_golden_triple_candidates_with_triple(
                triple=triple,
                golden_triples=golden_triples,
                new_golden_triple_candidates=new_golden_triple_candidates
            )

        return new_topic_entity_candidates, new_golden_triple_candidates

    def _get_subgraph_from_dois(self,
                                dois: List[str],
                                doi_to_entity_mapping: dict[str, Knowledge]) -> List[Subgraph] | None:
        """
        For each of the given dois, this method retrieves the corresponding
        subgraph from the knowledge graph. It returns a single subgraph
        containing all the triples from the publications related to the DOIs.

        Args:
            dois (List[str]): A list of DOIs for which to retrieve the subgraphs.
            doi_to_entity_mapping (dict[str, Knowledge]): A mapping of DOIs to
                their corresponding entities in the knowledge graph.

        Returns:
            List[Subgraph]: A list of subgraphs containing the triples from the
                publications related to the DOIs.
        """
        qa_pair_subgraph: List[Triple] = []
        for source_id in dois:
            source_subgraph = self._get_publication_subgraph(
                publication=doi_to_entity_mapping.get(source_id, None)
            )
            if not source_subgraph:
                return None
            qa_pair_subgraph.extend(source_subgraph.root)
        return qa_pair_subgraph

    def get_doi_to_entity_mapping(self,
                                  research_field_id: str) -> dict[str, Knowledge]:
        """
        This method prepares a mapping of DOIs to their corresponding entities
        in the knowledge graph. It retrieves the research field entity and
        finds the closest publications related to that research field. The
        mapping is stored in a dictionary where the DOI is the key and the
        corresponding entity is the value.

        Args:
            kg_config (KnowledgeGraphConfig): The configuration for the knowledge graph.
            research_field_id (str): The ID of the research field entity from which
                the publication should be mapped to their nodes.

        Returns:
            dict[str, Knowledge]: A dictionary mapping DOIs to their corresponding
                entities in the knowledge graph.
        """
        doi_to_entity_mapping = {}
        generator = ClusterBasedQuestionGenerator(
            graph=self.graph,
            cluster_options=None,
            llm_adapter=None,
            generator_options=ClusterGeneratorOptions(
                generation_options=None,
            )
        )
        research_field_entity = self.graph.get_entity_by_id(research_field_id)
        if not research_field_entity:
            raise ValueError(
                f"Research field entity with ID {research_field_id} not found.")

        publications = generator.get_closest_publications_from_topic(
            topic_entity=research_field_entity
        )

        for publication in publications:
            # we need to map them to their dois
            main_triple = self.graph.get_main_triple_from_publication(
                publication.uid)
            if main_triple:
                doi = main_triple.entity_object.text
                doi_to_entity_mapping[doi] = publication
            else:
                print(
                    f"Main triple not found for publication {publication.uid}.")
        return doi_to_entity_mapping

    def _get_publication_subgraph(self, publication: Knowledge) -> Subgraph | None:
        """
        This method retrieves the subgraph of a given publication entity from
        the knowledge graph.

        Args:
            publication (Knowledge): The publication entity for which to retrieve
                the subgraph.
            kg_config (KnowledgeGraphConfig): The configuration for the knowledge graph.

        Returns:
            Subgraph: The subgraph of the given publication entity or None if
                the publication is not found in the knowledge graph.
        """
        if not publication:
            return None
        subgraph_builder = SubgraphBuilder(knowledge_graph=self.graph)

        _, subgraph = subgraph_builder.get_subgraph(
            root_entity=publication,
            options=SubgraphOptions(
                go_against_direction=False,
                traverse_all_possible_edges=False
            )
        )

        if not subgraph:
            return None

        return subgraph

    def _save_updated_qa_dataset(self, qa_dataset_path: str, updated_qa_pairs: List[QAPair]):
        """
        Saves the updated QA dataset to a CSV file.

        Args:
            qa_dataset_path (str): The path to the original QA dataset.
            updated_qa_pairs (List[QAPair]): The list of updated QA pairs.
        """
        # Create the repaired dataset
        repaired_dataset = QADataset.from_qa_pairs(
            "repaired_dataset", updated_qa_pairs)
        folder = FilePathManager().get_file_directory(qa_dataset_path)
        updated_df = repaired_dataset.to_dataframe()
        # Read the original CSV to retrieve extra columns not in the QA model
        original_df = pd.read_csv(qa_dataset_path)

        # Identify columns that exist in the original CSV but not in the updated DataFrame
        missing_columns = set(original_df.columns) - set(updated_df.columns)
        if missing_columns:
            if "uid" not in updated_df.columns:
                raise ValueError(
                    "The updated QA dataframe does not contain 'uid' column for merging.")

            updated_df = updated_df.merge(
                original_df[list(missing_columns) + ["uid"]], on="uid", how="left")

        # Save the merged DataFrame as CSV
        if self.update_in_place:
            output_path = qa_dataset_path
        else:
            output_path = FilePathManager().combine_paths(folder, "updated_qa_dataset.csv")

        updated_df.to_csv(output_path, index=False)
        print(f"Repaired dataset saved to {output_path}")
