import pytest
from implementation.experimentation.evaluation.implementations.instruction_following_evaluator import InstructionFollowingEvaluator
from implementation.config.config_models import EvaluatorConfig
from core import LLMConfig

LLM_CONFIG = {
    "additional_params": {},
    "endpoint": "OpenAI",
    "name_model": "o3-mini",
    "temperature": None,
    "max_tokens": -1,
    "reasoning_effort": "low"
}


def test_correct_ranking():
    """Test the score method with correct ranking."""
    config = EvaluatorConfig(
        additional_params={
            "llm_config": LLMConfig.from_dict(LLM_CONFIG),
        },
        evaluator_type="instruction_following"
    )
    evaluator = InstructionFollowingEvaluator(config=config)
    output = {
        "initial_question": "Which papers have been published by the author Sten Grüner ranked in descending order by their publication year?",
        "generated_answer": """
            Based on the provided information, Sten Grüner has co-authored the following papers, ranked in descending order by publication year:
            1. A Comparison of MQTT Brokers for Distributed IoT Edge Computing (2020)
            2. A Four-Layer Architecture Pattern for Constructing and Managing Digital Twins (2019)
            """
    }
    golden_answer = "The publications by Sten Grüner, ranked in descending order of their publication year, are: 1. A Comparison of MQTT Brokers for Distributed IoT Edge Computing (2020), 2. A Four-Layer Architecture Pattern for Constructing and Managing Digital Twins (2019)."

    result = evaluator.score(output=output, golden_answer=golden_answer)

    assert result["instruction_following"] == 1.

    output = {
        "initial_question": "Among those papers that evaluate the property Reliability, what are the research objects that are evaluated? Rank the research objects in descending alphabetical order.",
        "generated_answer": "1. Reference Architecture, 2. Architecture Optimization Method, 3. Architecture Analysis Method, 4. Architectural Aspects."
    }
    golden_answer = "Among those papers that evaluate the property Reliability, the research objects that are evaluated, ranked in descending alphabetical order, are: Reference Architecture, Architecture Optimization Method, Architecture Analysis Method, and Architectural Aspects."

    result = evaluator.score(output=output, golden_answer=golden_answer)

    assert result["instruction_following"] == 1.
    
    output = {
        "initial_question": "Rank the evaluation method in descending order of their publication year.",
        "generated_answer": "The first is Interview which has been published in 2023. The second is Survey published in 2022. The last one is Case Study published in 2021."
    }
    
    golden_answer = "The evaluation methods, ranked in descending order of their publication year, are: 1. Interview (2023), 2. Survey (2022), 3. Case Study (2021)."
    
    result = evaluator.score(output=output, golden_answer=golden_answer)
    
    assert result["instruction_following"] == 1.


def test_wrong_ranking():
    """Test the score method with wrong ranking."""
    config = EvaluatorConfig(
        additional_params={
            "llm_config": LLM_CONFIG
        },
        evaluator_type="instruction_following"
    )
    evaluator = InstructionFollowingEvaluator(config=config)
    output = {
        "initial_question": "Which papers have been published by the author Sten Grüner ranked in descending order by their publication year?",
        "generated_answer": """
            Based on the provided information, Sten Grüner has co-authored the following papers, ranked in descending order by publication year:
            1. A Four-Layer Architecture Pattern for Constructing and Managing Digital Twins (2019)
            2. A Comparison of MQTT Brokers for Distributed IoT Edge Computing (2020)
            """
    }
    golden_answer = "The publications by Sten Grüner, ranked in descending order of their publication year, are: 1. A Comparison of MQTT Brokers for Distributed IoT Edge Computing (2020), 2. A Four-Layer Architecture Pattern for Constructing and Managing Digital Twins (2019)."

    result = evaluator.score(output=output, golden_answer=golden_answer)

    assert result["instruction_following"] == 0.

    output = {
        "initial_question": "Among those papers that evaluate the property Reliability, what are the research objects that are evaluated? Rank the research objects in descending alphabetical order.",
        "generated_answer": "1. Architectural Aspects, 2. Architecture Optimization Method, 3. Architecture Analysis Method, 4. Reference Architecture."
    }
    golden_answer = "Among those papers that evaluate the property Reliability, the research objects that are evaluated, ranked in descending alphabetical order, are: Reference Architecture, Architecture Optimization Method, Architecture Analysis Method, and Architectural Aspects."

    result = evaluator.score(output=output, golden_answer=golden_answer)

    assert result["instruction_following"] == 0.
    

def test_correct_counting():
    """Test the score method with correct counting."""
    config = EvaluatorConfig(
        additional_params={
            "llm_config": LLM_CONFIG
        },
        evaluator_type="instruction_following"
    )
    evaluator = InstructionFollowingEvaluator(config=config)
    output = {
        "initial_question": "How many papers have been published by the author Sten Grüner?",
        "generated_answer": "2 papers."
    }
    golden_answer = "Sten Grüner has co-authored 2 papers."

    result = evaluator.score(output=output, golden_answer=golden_answer)

    assert result["instruction_following"] == 1.
    
    output = {
        "initial_question": "How many research objects have been investigated in the paper?",
        "generated_answer": "The paper has four research objects that have been investigated."
    }
    golden_answer = "4 research objects"
    
    result = evaluator.score(output=output, golden_answer=golden_answer)
    
    assert result["instruction_following"] == 1.
    
def test_wrong_counting():
    """Test the score method with wrong counting."""
    config = EvaluatorConfig(
        additional_params={
            "llm_config": LLM_CONFIG
        },
        evaluator_type="instruction_following"
    )
    evaluator = InstructionFollowingEvaluator(config=config)
    output = {
        "initial_question": "How many research objects have been investigated in the paper?",
        "generated_answer": "The paper investigated Reference Architecture, Architecture Optimization Method, and Architecture Analysis Method."
    }
    golden_answer = "The paper has three research objects that have been investigated."

    result = evaluator.score(output=output, golden_answer=golden_answer)

    assert result["instruction_following"] == 0.

def test_no_instructions():
    """Test the score method with no instructions in this case, the evaluator should return 1.0."""
    config = EvaluatorConfig(
        additional_params={
            "llm_config": LLM_CONFIG
        },
        evaluator_type="instruction_following"
    )
    evaluator = InstructionFollowingEvaluator(config=config)
    output = {
        "initial_question": "Which research object has been investigated in the paper?",
        "generated_answer": "The paper has investigated the Reference Architecture."
    }
    golden_answer = "Reference architecture has been investigated in the paper."

    result = evaluator.score(output=output, golden_answer=golden_answer)

    assert result["instruction_following"] == 1.


if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
