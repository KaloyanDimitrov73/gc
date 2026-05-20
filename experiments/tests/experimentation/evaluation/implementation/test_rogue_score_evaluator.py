import pytest
from experimentation.evaluation.implementations.rouge_score_evaluator import RougeScoreEvaluator
from experimentation.config.evaluator_config import EvaluatorConfig

def test_rouge_score_evaluator():
    """Test whether the ROUGE score provides the expected results."""
    config = EvaluatorConfig(
        additional_params={},
        evaluator_type="rouge_score"
    )
    evaluator = RougeScoreEvaluator(config=config)
    output = {"generated_answer": "Deutschland wird wieder Weltmeister werden."}
    golden_answer = "Deutschland wird wieder Weltmeister werden."
    result = evaluator.score(output=output, golden_answer=golden_answer)
    
    assert result["rouge_1_f1"] == 1.0
    assert result["rouge_2_f1"] == 1.0
    assert result["rouge_L_f1"] == 1.0
    
    output = {"generated_answer": "Deutschland wird wieder Weltmeister werden."}
    golden_answer = "Deutschland wird Weltmeister"
    result = evaluator.score(output=output, golden_answer=golden_answer)
    
    assert result["rouge_1_f1"] < 1.0
    assert result["rouge_2_f1"] < 1.0
    assert result["rouge_L_f1"] < 1.0
    
    
    
if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])