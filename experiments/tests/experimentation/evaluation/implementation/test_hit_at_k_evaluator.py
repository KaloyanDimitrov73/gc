import pytest
from experimentation.evaluation.implementations.hit_at_k_evaluator import HitAtKEvaluator
from experimentation.config.evaluator_config import EvaluatorConfig
from core import Triple, Knowledge

TEST_TRIPLES = [
    Triple(
        entity_subject=Knowledge(text="Entity A", uid=""),
        entity_object=Knowledge(text="Entity B", uid=""),
        predicate="is related to"
    ),
    Triple(
        entity_subject=Knowledge(text="Entity C", uid=""),
        entity_object=Knowledge(text="Entity D", uid=""),
        predicate="is related to"
    )
]

def test_score_empty_input():
    """Test the score method with empty input."""
    config = EvaluatorConfig(additional_params={"k": 1, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    result = evaluator.score(output=None, golden_triples=None)
    assert result["hit@1_triples"] == 1.0

def test_score_empty_contexts():
    """Test the score method with empty contexts."""
    config = EvaluatorConfig(additional_params={"k": 1, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    result = evaluator.score(output={"contexts": []}, golden_triples=["test triple"])
    assert result["hit@1_triples"] == 0.0

def test_score_matching_triple():
    """Test the score method with a matching triple."""
    config = EvaluatorConfig(additional_params={"k": 1, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[0])},
            {"text": str(TEST_TRIPLES[1])}
        ]
    }
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["hit@1_triples"] == 1.0

def test_score_not_first():
    """Test the score method with a non-first matching triple."""
    config = EvaluatorConfig(additional_params={"k": 1, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[1])},
            {"text": str(TEST_TRIPLES[0])}
        ]
    }
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["hit@1_triples"] == 0.0

def test_score_multiple_golden_triples():
    """Test the score method with multiple golden triples."""
    config = EvaluatorConfig(additional_params={"k": 1, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "test triple 2"}
        ]
    }
    result = evaluator.score(
        output=output, 
        golden_triples=["test triple 1", "test triple 2"]
    )
    assert result["hit@1_triples"] == 0.5  

def test_score_with_k_2():
    """Test the score method with k=2."""
    config = EvaluatorConfig(additional_params={"k": 2, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "test triple 2"}
        ]
    }
    result = evaluator.score(
        output=output, 
        golden_triples=["test triple 1", "test triple 2"]
    )
    assert result["hit@2_triples"] == 1.0  

def test_entity_mode():
    """Test the score method in entity mode."""
    config = EvaluatorConfig(additional_params={"k": 2, "context_type": "entity"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    
    output = {
        "retrieved_context": [
            {"text": TEST_TRIPLES[0].entity_object.text},
            {"text": TEST_TRIPLES[0].entity_object.text}
        ]
    }
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["hit@2_entities"] == 0.5

def test_score_no_matches():
    """Test the score method with no matches."""
    config = EvaluatorConfig(additional_params={"k": 2, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    output = {"retrieved_context": [{"text": "irrelevant context"}]}
    result = evaluator.score(output=output, golden_triples=["non existent triple"])
    assert result["hit@2_triples"] == 0.0

def test_partial_hits_within_k():
    """Test the score method with partial hits within k."""
    config = EvaluatorConfig(additional_params={"k": 2, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "not relevant"},
            {"text": "test triple 3"}
        ]
    }
    golden = ["test triple 1", "test triple 2", "test triple 3"]
    result = evaluator.score(output=output, golden_triples=golden)
    assert result["hit@2_triples"] == pytest.approx(1/3)
    
def test_score_with_not_enough_contexts():
    """Test the score method with not enough contexts."""
    config = EvaluatorConfig(additional_params={"k": 5, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "test triple 2"}
        ]
    }
    result = evaluator.score(
        output=output, 
        golden_triples=["test triple 1", "test triple 2", "nonexistent triple"]
    )
    assert round(result["hit@5_triples"], 2) == round(2/3, 2)
    
def test_multiple_golden_at_k_1():
    """Test the score method with multiple golden triples at k=1."""
    config = EvaluatorConfig(additional_params={"k": 1, "context_type": "triple"},
                             evaluator_type="hit_at_k")
    evaluator = HitAtKEvaluator(config=config)
    
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "test triple 2"}
        ]
    }
    result = evaluator.score(output=output, golden_triples=["test triple 1", "test triple 2", "nonexistent triple"])
    assert round(result["hit@1_triples"], 2) == round(1/3, 2)
    
if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])