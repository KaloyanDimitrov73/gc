import pytest
from implementation.experimentation.evaluation.implementations.mrr_at_k_evaluator import MRRAtKEvaluator
from implementation.config.config_models import EvaluatorConfig
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
def mrr_at_1_evaluator():
    config = EvaluatorConfig(
        additional_params={"k": 1, "context_type": "triple"},
        evaluator_type="mrr_at_k"
    )
    return MRRAtKEvaluator(config=config)

@pytest.fixture
def mrr_at_1_evaluator_entity():
    config = EvaluatorConfig(
        additional_params={"k": 1, "context_type": "entity"},
        evaluator_type="mrr_at_k"
    )
    return MRRAtKEvaluator(config=config)

def test_mrr_score_empty_input(mrr_at_1_evaluator):
    """Test the score method with empty input."""
    # No golden triples and no output contexts
    result = mrr_at_1_evaluator.score(output=None, golden_triples=None)
    assert result["mrr@1_triples"] == 1.0

def test_mrr_score_empty_contexts(mrr_at_1_evaluator):
    """Test the score method with empty contexts."""
    # Empty contexts provided with a golden triple expected
    result = mrr_at_1_evaluator.score(output={"contexts": []}, golden_triples=["test triple"])
    assert result["mrr@1_triples"] == 0.0

def test_mrr_score_matching_first(mrr_at_1_evaluator):
    """Test the score method with a matching triple."""
    # First retrieved context matches a golden triple
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[0])},
            {"text": str(TEST_TRIPLES[1])}
        ]
    }
    result = mrr_at_1_evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["mrr@1_triples"] == 1.0

def test_mrr_score_not_first(mrr_at_1_evaluator):
    """Test the score method with a non-first matching triple."""
    # The first context does not match, so with k=1, no relevant context is found
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[1])},
            {"text": str(TEST_TRIPLES[0])}
        ]
    }
    result = mrr_at_1_evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    # Since k=1, only the first context is considered and it doesn't match.
    assert result["mrr@1_triples"] == 0.0

def test_mrr_score_with_k_2():
    """Test the score method with k=2 and a matching triple."""
    # Testing with k=2, expecting a match at the second position
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="mrr_at_k"
    )
    evaluator = MRRAtKEvaluator(config=config)
    
    output = {
        "retrieved_context": [
            {"text": "irrelevant context"},
            {"text": "test triple 1"}
        ]
    }
    # The first relevant triple is at rank 2
    result = evaluator.score(
        output=output, 
        golden_triples=["test triple 1"]
    )
    # Reciprocal rank of 2 is 0.5
    assert result["mrr@2_triples"] == 0.5

def test_mrr_entity_mode(mrr_at_1_evaluator_entity):
    """Test the score method in entity mode."""
    # Test evaluator in entity mode
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[0])},
            {"text": "some other entity"}
        ]
    }
    result = mrr_at_1_evaluator_entity.score(
        output=output,
        golden_triples=[str(TEST_TRIPLES[0])]
    )
    # Since the first retrieved context matches an entity from the golden triple,
    # reciprocal rank should be 1.0
    assert result["mrr@1_entities"] == 1.0

def test_mrr_score_no_matches(mrr_at_1_evaluator):
    """Test the score method with no matches."""
    # No retrieved contexts match the golden triples
    output = {"retrieved_context": [{"text": "irrelevant context"}]}
    result = mrr_at_1_evaluator.score(output=output, golden_triples=["non existent triple"])
    assert result["mrr@1_triples"] == 0.0

def test_mrr_partial_hit_within_k():
    """Test the score method with a partial hit within k."""
    # Testing partial hit not at the first position but within k
    config = EvaluatorConfig(
        additional_params={"k": 3, "context_type": "triple"},
        evaluator_type="mrr_at_k"
    )
    evaluator = MRRAtKEvaluator(config=config)
    
    output = {
        "retrieved_context": [
            {"text": "irrelevant context"},
            {"text": "another irrelevant context"},
            {"text": "test triple 3"}
        ]
    }
    result = evaluator.score(
        output=output,
        golden_triples=["test triple 3"]
    )
    # The first relevant triple is at rank 3, so reciprocal rank is 1/3
    assert pytest.approx(result["mrr@3_triples"], rel=1e-2) == 1/3

if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
