from sqa_system.core.data.models import QAPair
from .base.kg_qa_generation_strategy import KGQAGenerationStrategy



class QAGenerator:
    """
    Generates QA pairs based on a given strategy.
    """

    @staticmethod
    def run_generation(strategy: KGQAGenerationStrategy) -> list[QAPair]:
        """
        Generates QA pairs based on the given strategy.

        Args:
            strategy (KGQAGenerationStrategy): The strategy to use for generating QA pairs.

        Returns:
            list[QAPair]: The list of generated QA pairs.
        """
        return strategy.generate()
