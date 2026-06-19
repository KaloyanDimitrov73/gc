from sqa_system.core.data.models import Knowledge, Subgraph, QAPair, Triple
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph


from ...strategies import ClusterBasedQuestionGenerator, ClusterGeneratorOptions
from .qa_similarity_matcher import QASimilarityMatcher


class QAPairUpdater:
    """
    Class responsible for updating QAPairs.

    Args:
        graph (KnowledgeGraph): The knowledge graph to be used for updating QAPairs.
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.similarity_matcher = QASimilarityMatcher()

    def update_topic_entity(self,
                            old_qa_pair: QAPair,
                            updated_qa_pair: QAPair,
                            new_topic_entity_candidates: set[Knowledge]) -> Knowledge:
        """
        Updates a qa pair with the new topic entity. If there are multiple candidates,
        it will select the most matching one based on the topic entity value.

        Args:
            old_qa_pair (QAPair): The original QA Pair.
            updated_qa_pair (QAPair): The updated QA Pair.
            new_topic_entity_candidates (set[Knowledge]): A set of new topic
                entity candidates.
                
        Returns:
            Knowledge: The selected topic entity.
        """
        if len(new_topic_entity_candidates) > 1:
            
            topic_entity = QASimilarityMatcher().select_most_matching_candidate(
                candidates=list(new_topic_entity_candidates),
                target=old_qa_pair.topic_entity_value
            )
        else:
            topic_entity = new_topic_entity_candidates.pop()

        updated_qa_pair.topic_entity_value = topic_entity.text
        updated_qa_pair.topic_entity_id = topic_entity.uid
        return topic_entity

    def update_golden_tripels(self,
                              old_qa_pair: QAPair,
                              updated_qa_pair: QAPair,
                              new_golden_triple_candidates: dict[str, set[Triple]],
                              string_replacements: dict[str, str] = None):
        """
        Updates the golden triples of a qa pair with the new golden triples.
        If there are multiple candidates, it will select the most matching one
        based on the original golden triple.

        Args:
            old_qa_pair (QAPair): The original QA Pair.
            updated_qa_pair (QAPair): The updated QA Pair.
            new_golden_triple_candidates (dict[str, set]): A dictionary of new
                golden triple candidates.
            string_replacements (dict[str, str], optional): A dictionary of
                string replacements to be applied to the candidates before
                matching. Has to be in lower case. They key is the string to
                be replaced and the value is the string to replace it with.
        """
        original_golden_triples = old_qa_pair.golden_triples.copy()
        updated_golden_triples = []
        for orig_triple in original_golden_triples:
            if orig_triple in new_golden_triple_candidates:
                candidates = new_golden_triple_candidates[orig_triple]
                if len(candidates) > 1:
                    # Removing those candidates that have already been chosen, this is important
                    # if we have triples that are identical but their id is different
                    candidates = [
                        c for c in candidates if str(c) not in updated_golden_triples]
                    new_golden_triple = QASimilarityMatcher().select_most_matching_candidate(
                        candidates=list(candidates),
                        target=orig_triple,
                        string_replacements=string_replacements
                    )
                else:
                    new_golden_triple = candidates.pop()
                updated_golden_triples.append(str(new_golden_triple))
            else:
                raise ValueError(
                    f"Could not find candidate for golden triple: {orig_triple}")
        updated_qa_pair.golden_triples = updated_golden_triples

    def update_hop_amount(self,
                          updated_qa_pair: QAPair,
                          topic_entity: Knowledge,
                          qa_pair_subgraph: Subgraph,
                          updated_golden_triples: list[str]):
        """
        Updates the hop amount of the qa pair based on the new topic entity and
        the new golden triples.

        Args:
            updated_qa_pair (QAPair): The updated QA Pair.
            topic_entity (Knowledge): The new topic entity.
            qa_pair_subgraph (Subgraph): The subgraph of the QA Pair.
            updated_golden_triples (list[str]): The new golden triples.

        """
        generator = ClusterBasedQuestionGenerator(
            graph=self.graph,
            cluster_options=None,
            llm_adapter=None,
            generator_options=ClusterGeneratorOptions(
                generation_options=None,
            )
        )
        new_hop_amount = generator.calculate_hop_amount(
            subgraph=qa_pair_subgraph,
            root_entity=topic_entity,
            golden_triples=updated_golden_triples
        )
        updated_qa_pair.hops = new_hop_amount
