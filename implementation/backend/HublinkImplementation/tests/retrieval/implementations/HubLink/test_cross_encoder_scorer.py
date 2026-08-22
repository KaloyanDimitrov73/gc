from unittest.mock import MagicMock, patch

from hublink.retrieval.cross_encoder.cross_encoder_scorer import (
    CrossEncoderScorer,
)


@patch(
    "hublink.retrieval.cross_encoder.cross_encoder_scorer.CrossEncoder"
)
def test_cross_encoder_loads_weights_directly_onto_cpu(cross_encoder):
    model = MagicMock()
    cross_encoder.return_value = model

    scorer = CrossEncoderScorer("test-model")

    cross_encoder.assert_called_once_with(
        "test-model",
        model_kwargs={"device_map": "cpu"},
    )
    assert scorer.model is model


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
