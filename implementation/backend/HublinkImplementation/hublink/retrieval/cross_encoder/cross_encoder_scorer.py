from threading import Lock
from typing import List

from sentence_transformers import CrossEncoder

from core.logging.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_MODEL_CACHE: dict[str, CrossEncoder] = {}
_MODEL_PREDICTION_LOCKS: dict[str, Lock] = {}
_MODEL_CACHE_LOCK = Lock()


def _get_or_load_model(model_name: str) -> tuple[CrossEncoder, Lock]:
    """Load each cross-encoder once and share it between retrieval requests."""
    with _MODEL_CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            logger.info("Loading cross-encoder model: %s", model_name)
            _MODEL_CACHE[model_name] = CrossEncoder(
                model_name,
                device="cpu",
            )
            _MODEL_PREDICTION_LOCKS[model_name] = Lock()

        return _MODEL_CACHE[model_name], _MODEL_PREDICTION_LOCKS[model_name]


class CrossEncoderScorer:
    """
    Wraps a sentence-transformers CrossEncoder model to score how relevant
    a batch of texts is to a question.
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        self.model_name = model_name
        # Weave evaluates several questions concurrently. Loading in this
        # process-wide cache prevents concurrent Hugging Face/Accelerate model
        # initialization and avoids keeping one model copy per question.
        self.model, self._prediction_lock = _get_or_load_model(model_name)

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
        # CrossEncoder.predict() calls Module.to(device), so protect the shared
        # model from concurrent mutation while Weave runs multiple questions.
        with self._prediction_lock:
            scores = self.model.predict(pairs)
        return [float(score) for score in scores]
