from typing import List, Optional
from .base.config import Config
from .pipeline_config import PipelineConfig
from implementation.models.parameter_range import ParameterRange
from .dataset_config import DatasetConfig
from .evaluator_config import EvaluatorConfig


class ExperimentConfig(Config):
    """
    Configuration class for an experiment.
    """
    base_pipeline_config: PipelineConfig
    parameter_ranges: List[ParameterRange]
    evaluators: List[EvaluatorConfig]
    qa_dataset: Optional[DatasetConfig] = None

    def generate_name(self) -> str:
        ranges_str = ""
        for p_range in self.parameter_ranges:
            ranges_str += f"{p_range.config_name}_{p_range.parameter_name}"
        if self.qa_dataset:
            return (f"{self.base_pipeline_config.name}_"
                    f"{self.qa_dataset.name}_{ranges_str}")

        return (f"{self.base_pipeline_config.name}_"
                f"{ranges_str}")
