from typing import Dict

from sentence_transformers import SparseEncoder

from core.logging.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-selfdistil"

_encoder_cache: Dict[str, SparseEncoder] = {}


def get_splade_encoder(
        model_name: str = DEFAULT_SPLADE_MODEL) -> SparseEncoder:
    """Returns a process-wide cached Sentence Transformers sparse encoder."""
    if model_name not in _encoder_cache:
        logger.info("Loading SPLADE model: %s", model_name)
        _encoder_cache[model_name] = SparseEncoder(model_name, device="cpu")
    return _encoder_cache[model_name]
