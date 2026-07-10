import pytest
from implementation.config.config_models import ExperimentConfig
from implementation.config.config_models import PipelineConfig
from implementation.config.data_models.parameter_range import ParameterRange
from implementation.config.config_models import GenerationConfig
from language_model.config.llm_config import LLMConfig
from implementation.core import DatasetConfig
from implementation.experimentation.experiment_runner import ExperimentRunner

@pytest.fixture
def experiment_config():
    test_generation = GenerationConfig(
                           name="Test Generation",
                           llm_config=LLMConfig(
                               name="Test LLM",
                               endpoint="openai",
                               name_model="gpt-4o-mini",
                               temperature=0.0,
                               max_tokens=100
                           )
                       )
    base_pipeline = PipelineConfig(
        name="Test Pipeline",
        pipes=[test_generation]
    )
    dataset_config = DatasetConfig(
        name="Test Dataset",
        file_name="question_answering_codereview.csv",
        loader="CSVQALoader",
        loader_limit=-1
    )
    param_range = ParameterRange(
        config_name="Test LLM",
        parameter_name="temperature",
        values=[0.5, 0.7, 0.9]
    )
    return ExperimentConfig(
        name="Test Experiment",
        base_pipeline_config = base_pipeline,
        parameter_ranges=[
            param_range           
            ],
        evaluators=[],
        qa_dataset=dataset_config
    )

@pytest.fixture
def experiment_runner(experiment_config):
    return ExperimentRunner(experiment_config)

def test_generate_pipeline_configs(experiment_runner):
    """Test whether the generation of pipeline configs works as expected."""
    configs = list(experiment_runner.experiment_preparer.prepare_pipeline_configs())
    assert isinstance(configs, list)
    # based on the given experiment configuration we expect 
    # the following to be true
    assert len(configs) == 4   
    assert configs[0].pipes[0].llm_config.temperature == 0.0
    assert configs[1].pipes[0].llm_config.temperature == 0.5
    assert configs[2].pipes[0].llm_config.temperature == 0.7
    assert configs[3].pipes[0].llm_config.temperature == 0.9

if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])