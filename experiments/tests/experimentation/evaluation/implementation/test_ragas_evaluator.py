import pytest
from experimentation.evaluation.implementations.ragas_evaluator.ragas_evaluator import RagasEvaluator
from experimentation.config.evaluator_config import EvaluatorConfig
from core import Triple, Knowledge

LLM_CONFIG = {
    "additional_params": {},
    "endpoint": "OpenAI",
    "name_model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": -1
}

EMBEDDING_MODEL_CONFIG = {
    "additional_params": {},
    "endpoint": "OpenAI",
    "name_model": "text-embedding-3-small"
}

EVALUATOR_CONFIG = EvaluatorConfig(
    additional_params={
        "llm": LLM_CONFIG,
        "embedding_model": EMBEDDING_MODEL_CONFIG
    },
    evaluator_type="ragas"
)


def test_faithfulness():
    """Test whether the faitfulness score provides the expected results."""
    copied_config = EVALUATOR_CONFIG.model_copy()
    copied_config.additional_params["metrics"] = ["faithfulness"]

    evaluator = RagasEvaluator(config=copied_config)
    output = {
        "initial_question": "In which venue has the paper with the title 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' been published?",
        "generated_answer": "The paper titled ""Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models"" was published at the International Conference on Software Architecture (ICSA)",
        "retrieved_context": ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']
    }
    golden_answer = "The paper 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' has been published at the International Conference on Software Architecture (ICSA)."
    golden_triples = ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']

    result = evaluator.score(
        output=output, golden_answer=golden_answer, golden_triples=golden_triples)

    assert result["faithfulness"] == 1.

    output = {
        "initial_question": "In which venue has the paper with the title 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' been published?",
        "generated_answer": "The paper titled ""Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models"" has not been published yet.",
        "retrieved_context": ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']
    }
    golden_answer = "The paper 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' has been published at the International Conference on Software Architecture (ICSA)."
    golden_triples = ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']

    result = evaluator.score(
        output=output, golden_answer=golden_answer, golden_triples=golden_triples)

    assert result["faithfulness"] < 1.


def test_factual_correctness_f1():
    """Test whether the factual correctness f1 score provides the expected results."""
    copied_config = EVALUATOR_CONFIG.model_copy()
    copied_config.additional_params["metrics"] = ["factual_correctness_f1"]

    evaluator = RagasEvaluator(config=copied_config)
    output = {
        "initial_question": "In which venue has the paper with the title 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' been published?",
        "generated_answer": "The paper titled ""Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models"" was published at the International Conference on Software Architecture (ICSA)",
        "retrieved_context": ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']
    }
    golden_answer = "The paper 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' has been published at the International Conference on Software Architecture (ICSA)."
    golden_triples = ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']

    result = evaluator.score(
        output=output, golden_answer=golden_answer, golden_triples=golden_triples)

    assert result["factual_correctness(mode=f1)"] == 1.

    output = {
        "initial_question": "In which venue has the paper with the title 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' been published?",
        "generated_answer": "The paper titled Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models has not been published yet.",
        "retrieved_context": ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']
    }

    golden_answer = "The paper 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' has been published at the International Conference on Software Architecture (ICSA)."
    golden_triples = ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']

    result = evaluator.score(
        output=output, golden_answer=golden_answer, golden_triples=golden_triples)

    assert result["factual_correctness(mode=f1)"] < 1.
    
    
    output = {
        "initial_question": "What evaluation methods have been applied to evaluate accuracy on the investigated objects in papers published by Duc Le?",
        "generated_answer": "Based on the combined evidence, the papers published by Duc Le have evaluated accuracy on the investigated objects by applying benchmarking techniques.",
        "retrieved_context": ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']
    }

    golden_answer = "The evaluation methods that have been applied to evaluate accuracy on the investigated objects in papers published by Duc Le are Technical Experiment and Data Science."
    golden_triples = ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']

    result = evaluator.score(
        output=output, golden_answer=golden_answer, golden_triples=golden_triples)

    assert result["factual_correctness(mode=f1)"] < 1.

def test_response_relevancy():
    """Test whether the response relevancy score provides the expected results."""
    copied_config = EVALUATOR_CONFIG.model_copy()
    copied_config.additional_params["metrics"] = ["response_relevancy"]

    evaluator = RagasEvaluator(config=copied_config)
    output = {
        "initial_question": "In which venue has the paper with the title 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' been published?",
        "generated_answer": "The paper titled ""Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models"" was published at the International Conference on Software Architecture (ICSA)",
        "retrieved_context": ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']
    }
    golden_answer = "The paper 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' has been published at the International Conference on Software Architecture (ICSA)."
    golden_triples = ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']

    result = evaluator.score(
        output=output, golden_answer=golden_answer, golden_triples=golden_triples)

    assert result["answer_relevancy"] > 0.9

    output = {
        "initial_question": "In which venue has the paper with the title 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' been published?",
        "generated_answer": "I dont know.",
        "retrieved_context": ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']
    }

    golden_answer = "The paper 'Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models' has been published at the International Conference on Software Architecture (ICSA)."
    golden_triples = ['(R872741:Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models, venue, R820814:International Conference on Software Architecture (ICSA))']

    result = evaluator.score(
        output=output, golden_answer=golden_answer, golden_triples=golden_triples)

    assert result["answer_relevancy"] < 0.3


if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
