from hublink_shared.config import (
    AdditionalConfigParameter,
    Config,
    DatasetConfig,
    RestrictionType,
)
from implementation.config.config_models import (
    AdditionalConfigParameter as ExperimentAdditionalConfigParameter,
    Config as ExperimentConfig,
    DatasetConfig as ExperimentDatasetConfig,
    ExperimentConfig as ExperimentRunConfig,
    GenerationConfig,
    LLMConfig,
    PipelineConfig,
    RestrictionType as ExperimentRestrictionType,
)


def test_experiment_config_package_reexports_shared_types():
    assert ExperimentAdditionalConfigParameter is AdditionalConfigParameter
    assert ExperimentConfig is Config
    assert ExperimentDatasetConfig is DatasetConfig
    assert ExperimentRestrictionType is RestrictionType


def test_experiment_models_accept_shared_config_types():
    llm_config = LLMConfig(
        endpoint="openai",
        name_model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=100,
    )
    pipeline_config = PipelineConfig(
        pipes=[GenerationConfig(llm_config=llm_config)],
    )
    dataset_config = DatasetConfig(
        file_name="questions.csv",
        loader="CSVQALoader",
        loader_limit=-1,
    )

    experiment_config = ExperimentRunConfig(
        base_pipeline_config=pipeline_config,
        parameter_ranges=[],
        evaluators=[],
        qa_dataset=dataset_config,
    )

    assert isinstance(experiment_config, Config)
    assert experiment_config.qa_dataset is dataset_config
