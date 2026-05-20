from abc import ABC
from .kg_qa_generation_strategy import KGQAGenerationStrategy


class SubgraphStrategy(KGQAGenerationStrategy, ABC):
    """
    Strategy to generate a subgraph from a topic entity and then generate questions and answers
    """
