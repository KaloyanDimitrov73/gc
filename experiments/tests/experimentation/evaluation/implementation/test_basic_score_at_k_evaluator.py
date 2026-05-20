import pytest
from experimentation.evaluation.implementations.basic_score_evaluator import BasicScoreEvaluator
from experimentation.config.evaluator_config import EvaluatorConfig
from sqa_system.core.data.models import Triple, Knowledge

TEST_TRIPLES = [
    Triple(
        entity_subject=Knowledge(text="Entity A", uid="0"),
        entity_object=Knowledge(text="Entity B", uid="1"),
        predicate="is related to"
    ),
    Triple(
        entity_subject=Knowledge(text="Entity C", uid="2"),
        entity_object=Knowledge(text="Entity D", uid="3"),
        predicate="is related to"
    )
]


def test_empty_input():
    """Test the score method with empty input."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="basic_score_at_k"
    )
    evaluator = BasicScoreEvaluator(config=config)
    
    result = evaluator.score(output=None, golden_triples=None)
    # When we have no output and no golden data than the retriever 
    # correctly did not retrieve any data
    precision = result["precision@2_triples"]
    recall = result["recall@2_triples"]
    f1 = result["f1@2_triples"]
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0

def test_empty_contexts():
    """Test the score method with empty contexts."""
    config = EvaluatorConfig(
        additional_params={"k": 2, "context_type": "triple"},
        evaluator_type="basic_score_at_k"
    )
    evaluator = BasicScoreEvaluator(config=config)
    
    result = evaluator.score(output={"contexts": []}, golden_triples=["test triple"])
    # With empty contexts, precision, recall and f1 should all be 0.0.
    precision = result["precision@2_triples"]
    recall = result["recall@2_triples"]
    f1 = result["f1@2_triples"]
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0

def test_match_k1():
    """Test the score method with a matching triple."""
    config = EvaluatorConfig(
        additional_params={"k": 1, "context_type": "triple"},
        evaluator_type="basic_score_at_k"
    )
    evaluator = BasicScoreEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[0])}
        ]
    }
    # Single correct match within top-1.
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["precision@1_triples"] == 1.0
    assert result["recall@1_triples"] == 1.0
    assert result["f1@1_triples"] == 1.0

    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[1])},
            {"text": "irrelevant triple"}
        ]
    }
    # Only the first context is considered due to k=1.
    # Golden contains two triples, but top context matches one of them.
    result = evaluator.score(
        output=output, 
        golden_triples=[str(TEST_TRIPLES[0]), str(TEST_TRIPLES[1])]
    )
    # Since k=1 and the first retrieved triple is a correct match:
    # tp = 1, fp = 0, fn = 1 => precision=1.0, recall=0.5, f1 ≈ 0.667
    assert result["precision@1_triples"] == 1.0
    assert pytest.approx(result["recall@1_triples"], rel=1e-2) == 0.5
    assert pytest.approx(result["f1@1_triples"], rel=1e-2) == 2 * (1.0 * 0.5) / (1.0 + 0.5)
    
def test_match_k5():
    """Test the score method with k=5."""
    config = EvaluatorConfig(
        additional_params={"k": 5, "context_type": "triple"},
        evaluator_type="basic_score_at_k"
    )
    evaluator = BasicScoreEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": str(TEST_TRIPLES[0])},
            {"text": "irrelevant triple"}
        ]
    }
    # Only one correct match within top-5.
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    # Since k=5 and the first retrieved triple is a correct match:
    # tp = 1, fp = 1, fn = 0 => precision=0.5, recall=1.0, f1=0.667
    assert result["precision@5_triples"] == 0.5
    assert result["recall@5_triples"] == 1.0
    assert result["f1@5_triples"] == round(2 * (0.5 * 1.0) / (0.5 + 1.0), 2)

def test_no_matches():
    """Test the score method with no matches."""
    config = EvaluatorConfig(
        additional_params={"k": 1, "context_type": "triple"},
        evaluator_type="basic_score_at_k"
    )
    evaluator = BasicScoreEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": "irrelevant triple"}
        ]
    }
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    # No correct matches: tp = 0, fp = 1, fn = 1 => precision = 0.0, recall = 0.0, f1 = 0.0
    assert result["precision@1_triples"] == 0.0
    assert result["recall@1_triples"] == 0.0
    assert result["f1@1_triples"] == 0.0

def test_entity_mode():
    """Test the score method in entity mode."""
    config = EvaluatorConfig(
        additional_params={"k": 5, "context_type": "entity"},
        evaluator_type="basic_score_at_k"
    )
    evaluator = BasicScoreEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": str(Triple(
                entity_subject=Knowledge(text="irrelevant"),
                predicate="not relevant",
                entity_object=Knowledge(text="Entity B", uid="1")
            ))},
            {"text": str(Triple(
                entity_subject=Knowledge(text="Entity A", uid="0"),
                predicate="not relevant",
                entity_object=Knowledge(text="irrelevant")
            ))}
        ]
    }
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["precision@5_entities"] == 0.5
    assert result["recall@5_entities"] == 1.0
    assert result["f1@5_entities"] == round(2 * (0.5 * 1.0) / (0.5 + 1.0), 2)
    
    config = EvaluatorConfig(
        additional_params={"k": 5, "context_type": "triple"},
        evaluator_type="basic_score_at_k"
    )
    evaluator = BasicScoreEvaluator(config=config)
    output = {
        "retrieved_context": [
            {"text": str(Triple(
                entity_subject=Knowledge(text="irrelevant"),
                predicate="not relevant",
                entity_object=Knowledge(text="Entity B", uid="1")
            ))}
        ]
    }
    result = evaluator.score(output=output, golden_triples=[str(TEST_TRIPLES[0])])
    assert result["precision@5_triples"] == 0.0
    assert result["recall@5_triples"] == 0.0
    assert result["f1@5_triples"] == 0.0

if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
