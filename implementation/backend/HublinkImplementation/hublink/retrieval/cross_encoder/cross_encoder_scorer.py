from typing import List

from sentence_transformers import CrossEncoder

from core.logging.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderScorer:
    """
    Wraps a sentence-transformers CrossEncoder model to score how relevant
    a batch of texts is to a question.
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        self.model_name = model_name
        logger.info("Loading cross-encoder model: %s", model_name)
        # Load weights directly onto the CPU instead of loading first and then
        # calling Module.to("cpu"). The latter fails if Transformers leaves any
        # parameter on the meta device during its low-memory loading path.
        # Keeping the cross-encoder on CPU also avoids competing with the
        # LLM/embedding model for GPU memory.
        self.model = CrossEncoder(
            model_name,
            model_kwargs={"device_map": "cpu"},
        )

    def score_batch(self, question: str, path_texts: List[str]) -> List[float]:
        """
        Scores each path_text for how relevant it is to the question.

        Args:
            question (str): The question.
            path_texts (List[str]): The textual descriptions of the HubPaths
                to score, in the order the scores should be returned in.

        Returns:
            List[float]: The relevance score for each path_text, in the same
                order as path_texts.
        """
        pairs = [[question, path_text] for path_text in path_texts]
        return [float(score) for score in self.model.predict(pairs)]
