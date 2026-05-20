import pytest
from experimentation.evaluation.implementations.exact_match_evaluator import ExactMatchEvaluator
from experimentation.config.evaluator_config import EvaluatorConfig
from sqa_system.core.data.models import Triple, Knowledge

# Sample triple for entity-mode tests if needed
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

def test_exact_match_empty_input():
    """Test the score method with empty input."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="exact_match_at_k"
    )
    evaluator = ExactMatchEvaluator(config=config)
    result = evaluator.score(output=None, golden_triples=None)
    # When no output and no golden data are provided, Expect EM of 1.0
    assert result["exact_match@2_triples"] == 1.0

def test_exact_match_empty_contexts():
    """Test the score method with empty contexts."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="exact_match_at_k"
    )
    evaluator = ExactMatchEvaluator(config=config)
    result = evaluator.score(output={"contexts": []}, golden_triples=["test triple"])
    # With empty contexts but non-empty golden, EM should be 0.0
    assert result["exact_match@2_triples"] == 0.0

def test_exact_match_matching_triple():
    """Test the score method with a matching triple."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="exact_match"
    )
    evaluator = ExactMatchEvaluator(config=config)
    output = {
        "retrieved_context": [{"text": str(TEST_TRIPLES[0])}]
    }
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    # The retrieved context exactly matches the golden triple.
    assert result["exact_match@2_triples"] == 1.0

def test_exact_match_not_matching():
    """Test the score method with a non-matching triple."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="exact_match_at_k"
    )
    evaluator = ExactMatchEvaluator(config=config)
    output = {"retrieved_context": [{"text": str(TEST_TRIPLES[0])}]}
    result = evaluator.score(output=output, golden_triples=["some triple"])
    # The retrieved context does not match any golden triple.
    assert result["exact_match@2_triples"] == 0.0

def test_exact_match_with_k_greater_than_contexts():
    """Test the score method with k greater than the number of contexts."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="exact_match"
    )
    evaluator = ExactMatchEvaluator(config=config)
    output = {
        "retrieved_context": [{"text": str(TEST_TRIPLES[0])}]
    }
    # Since k=2 but only one context is provided, we only consider the one available.
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0]), str(TEST_TRIPLES[1])])
    # one exact match out of one retrieved context -> score 1.0
    assert result["exact_match@2_triples"] == 1.0

def test_exact_match_partial_matches():
    """Test the score method with partial matches."""
    config = EvaluatorConfig(
        additional_params={"k": 3, "context_type": "triple"},
        evaluator_type="exact_match"
    )
    evaluator = ExactMatchEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": "match1"},
            {"text": "no match"},
            {"text": "match2"}
        ]
    }
    # Only first 3 contexts considered, with two exact matches expected.
    result = evaluator.score(output=output, golden_triples=["match1", "match2"])
    # 2 exact matches out of 3 contexts -> 2/3 ≈ 0.6667
    assert pytest.approx(result["exact_match@3_triples"], rel=1e-2) == 2/3

def test_exact_match_entity_mode():
    """Test the score method in entity mode."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "entity"},
        evaluator_type="exact_match"
    )
    evaluator = ExactMatchEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": str(Triple(
                entity_subject=Knowledge(text="irrelevant", uid=""),
                predicate="not relevant",
                entity_object=Knowledge(text="Entity B", uid="")
            ))}
        ]
    }
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[1]), str(TEST_TRIPLES[0])])
    assert result["exact_match@2_entities"] == 0.5

if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
