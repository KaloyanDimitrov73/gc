import pytest
from implementation.experimentation.evaluation.implementations.map_at_k_evaluator import MAPAtKEvaluator
from implementation.config.models import EvaluatorConfig
from core import Triple, Knowledge

# Sample triples for testing
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

@pytest.fixture
def map_at_1_evaluator():
    config = EvaluatorConfig(
        additional_params={"k": 1, "context_type": "triple"},
        evaluator_type="map_at_k"
    )
    return MAPAtKEvaluator(config=config)

@pytest.fixture
def map_at_1_evaluator_entity():
    config = EvaluatorConfig(
        additional_params={"k": 1, "context_type": "entity"},
        evaluator_type="map_at_k"
    )
    return MAPAtKEvaluator(config=config)

def test_map_score_empty_input(map_at_1_evaluator):
    """Test the score method with empty input."""
    result = map_at_1_evaluator.score(output=None, golden_triples=None)
    assert result["map@1_triples"] == 1.0

def test_map_score_empty_contexts(map_at_1_evaluator):
    """Test the score method with empty contexts."""
    result = map_at_1_evaluator.score(output={"contexts": []}, golden_triples=["test triple"])
    assert result["map@1_triples"] == 0.0

def test_map_score_matching_triple(map_at_1_evaluator):
    """Test the score method with a matching triple."""
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[0])},
            {"text": str(TEST_TRIPLES[1])}
        ]
    }
    result = map_at_1_evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["map@1_triples"] == 1.0

def test_map_score_not_first(map_at_1_evaluator):
    """Test the score method with a non-first matching triple."""
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[1])},
            {"text": str(TEST_TRIPLES[0])}
        ]
    }
    result = map_at_1_evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["map@1_triples"] == 0.0

def test_map_score_multiple_golden_triples(map_at_1_evaluator):
    """Test the score method with multiple golden triples."""
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "test triple 2"}
        ]
    }
    result = map_at_1_evaluator.score(
        output=output, 
        golden_triples=["test triple 1", "test triple 2"]
    )
    # For k=1, only the first context is considered even if multiple golden triples exist.
    # Since the first context matches one of the golden triples, MAP@1 will be 1.0.
    assert result["map@1_triples"] == 1.0  

def test_map_score_with_k_2():
    """Test the score method with k=2 and multiple golden triples."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="map_at_k"
    )
    evaluator = MAPAtKEvaluator(config=config)
    
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
    # Both retrieved contexts match distinct golden triples within top 2.
    # For MAP@2, sum_precisions = 1/1 + 2/2 = 2, denominator = min(2,2) = 2, so MAP = 1.0
    assert result["map@2_triples"] == 1.0  

def test_map_entity_mode(map_at_1_evaluator_entity):
    """Test the score method in entity mode."""
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[0])},
            {"text": str(TEST_TRIPLES[1])}
        ]
    }
    result = map_at_1_evaluator_entity.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["map@1_entities"] == 1.0

def test_map_score_no_matches(map_at_1_evaluator):
    """Test the score method with no matches."""
    output = {"retrieved_context": [{"text": "irrelevant context"}]}
    result = map_at_1_evaluator.score(output=output, golden_triples=["non existent triple"])
    assert result["map@1_triples"] == 0.0

def test_map_partial_hits_within_k(map_at_1_evaluator):
    """Test the score method with partial hits within k."""
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "not relevant"},
            {"text": "test triple 3"}
        ]
    }
    golden = ["test triple 1", "test triple 2", "test triple 3"]
    result = map_at_1_evaluator.score(output=output, golden_triples=golden)
    # For MAP@1, only the first context is used. Since it matches one golden triple,
    # sum_precisions = 1/1 = 1 and denominator = 1, so the result is 1.0.
    assert result["map@1_triples"] == 1.0

def test_map_score_with_not_enough_contexts():
    """Test the score method with not enough contexts."""
    config = EvaluatorConfig(
        additional_params={"k": 5, "context_type": "triple"},
        evaluator_type="map_at_k"
    )
    evaluator = MAPAtKEvaluator(config=config)
    
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "test triple 2"}
        ]
    }
    # golden contains 3 triples, but only 2 are retrieved.
    # For k=5, denominator = min(3,5) = 3.
    # First context matches: precision = 1/1 = 1.
    # Second context matches: precision = 2/2 = 1.
    # Sum_precisions = 2, MAP = 2/3 ≈ 0.6667.
    result = evaluator.score(
        output=output, 
        golden_triples=["test triple 1", "test triple 2", "nonexistent triple"]
    )
    assert pytest.approx(result["map@5_triples"], rel=1e-2) == 2/3

def test_multiple_golden_at_k_1():
    """Test the score method with multiple golden triples at k=1."""
    config = EvaluatorConfig(
        additional_params={"k": 1, "context_type": "triple"},
        evaluator_type="map_at_k"
    )
    evaluator = MAPAtKEvaluator(config=config)
    
    output = {
        "retrieved_context": [
            {"text": "test triple 1"},
            {"text": "test triple 2"}
        ]
    }
    # With k=1, only the first retrieved context is considered.
    # Regardless of how many golden triples there are, the result should be 1.0 if the first context matches one.
    result = evaluator.score(
        output=output, 
        golden_triples=["test triple 1", "test triple 2", "nonexistent triple"]
    )
    assert result["map@1_triples"] == 1.0

if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
