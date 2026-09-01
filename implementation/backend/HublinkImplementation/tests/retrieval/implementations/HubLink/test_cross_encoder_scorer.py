from unittest.mock import MagicMock, patch

import pytest

from hublink.retrieval.cross_encoder import cross_encoder_scorer
from hublink.retrieval.cross_encoder.cross_encoder_scorer import (
    CrossEncoderScorer,
)


@pytest.fixture(autouse=True)
def clear_cross_encoder_cache():
    with cross_encoder_scorer._MODEL_CACHE_LOCK:
        cross_encoder_scorer._MODEL_CACHE.clear()
        cross_encoder_scorer._MODEL_PREDICTION_LOCKS.clear()
    yield
    with cross_encoder_scorer._MODEL_CACHE_LOCK:
        cross_encoder_scorer._MODEL_CACHE.clear()
        cross_encoder_scorer._MODEL_PREDICTION_LOCKS.clear()


@patch(
    "hublink.retrieval.cross_encoder.cross_encoder_scorer.CrossEncoder"
)
def test_cross_encoder_loads_onto_cpu(cross_encoder):
    model = MagicMock()
    cross_encoder.return_value = model

    scorer = CrossEncoderScorer("test-model")

    cross_encoder.assert_called_once_with(
        "test-model",
        device="cpu",
    )
    assert scorer.model is model


@patch(
    "hublink.retrieval.cross_encoder.cross_encoder_scorer.CrossEncoder"
)
def test_cross_encoder_model_is_reused(cross_encoder):
    model = MagicMock()
    cross_encoder.return_value = model

    first_scorer = CrossEncoderScorer("test-model")
    second_scorer = CrossEncoderScorer("test-model")

    cross_encoder.assert_called_once_with("test-model", device="cpu")
    assert first_scorer.model is model
    assert second_scorer.model is model


@patch(
    "hublink.retrieval.cross_encoder.cross_encoder_scorer.CrossEncoder"
)
def test_score_batch_returns_plain_floats(cross_encoder):
    cross_encoder.return_value.predict.return_value = [1, 0.25]
    scorer = CrossEncoderScorer("test-model")

    scores = scorer.score_batch("question", ["first", "second"])

    cross_encoder.return_value.predict.assert_called_once_with(
        [["question", "first"], ["question", "second"]]
    )
    assert scores == [1.0, 0.25]
