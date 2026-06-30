import pytest
from implementation.experimentation.evaluation.implementations.bleu_score_evaluator import BleuScoreEvaluator
from implementation.config.models import EvaluatorConfig

def test_bleu_score_evaluator():
    """Test whether the BLEU score provides the expected results."""
    config = EvaluatorConfig(
        additional_params={},
        evaluator_type="bleu_score"
    )
    evaluator = BleuScoreEvaluator(config=config)
    output = {"generated_answer": "Deutschland wird wieder Weltmeister werden."}
    golden_answer = "Deutschland wird wieder Weltmeister werden."
    result = evaluator.score(output=output, golden_answer=golden_answer)
    
    assert result["bleu_score"] == 1.0
    
    output = {"generated_answer": "Deutschland wird wieder Weltmeister werden."}
    golden_answer = "Deutschland wird Weltmeister"
    result = evaluator.score(output=output, golden_answer=golden_answer)
    
    assert result["bleu_score"] < 1.0
    
if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])