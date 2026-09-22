from typing import List, Type
import json

from sqa_system.core.data.models import ParameterRange
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.config.models import (
    Config,
    ExperimentConfig,
    PipelineConfig,
    DatasetConfig,
    EvaluatorConfig,
    KnowledgeGraphConfig
)
from sqa_system.core.config.config_manager import ConfigManagerFactory


class ExperimentConfigBuilder:
    """
    Class to build experiment configurations based on a baseline, evaluators,
    qa dataset and parameter ranges. This allows to distribute the parts of a
    experiment across multiple files and then combine them into a
    single experiment configuration.
    """

    def __init__(self):
        self.baseline: PipelineConfig = None
        self.parameter_ranges: List[ParameterRange] = []
        self.evaluators: List[EvaluatorConfig] = []
        self.qa_dataset: DatasetConfig = None
        self.config_manager_factory = ConfigManagerFactory()
        self.fpm = FilePathManager()

    def build(self) -> ExperimentConfig:
        """
        Builds the experiment configuration based on the before set parameters.

        Returns:
            ExperimentConfig: The generated experiment configuration.
        """
        if self.baseline is None:
            raise ValueError("Baseline pipeline config is not set")
        if len(self.evaluators) == 0:
            raise ValueError("Evaluator configs are not set")
        return ExperimentConfig(
            base_pipeline_config=self.baseline,
            parameter_ranges=self.parameter_ranges,
            evaluators=self.evaluators,
            qa_dataset=self.qa_dataset
        )
    
    def set_knowledge_graph_by_path(self, knowledge_graph_path: str):
        """
        Allows to set or change the knowledge graph config in the baseline.
        
        Args:
            knowledge_graph_path (str): The path to the knowledge graph config file.
        """
        if not self.fpm.file_path_exists(knowledge_graph_path):
            raise ValueError(f"File {knowledge_graph_path} does not exist")
        loaded_configs = self.load_configs_from_path(
            knowledge_graph_path, KnowledgeGraphConfig)
        if len(loaded_configs) > 1:
            print("Found multiple pipeline configs, using the first one")
        knowledge_graph_config = loaded_configs[0]
        
        for pipe in self.baseline.pipes:
            if pipe.type == "kg_retrieval":
                pipe.knowledge_graph_config = knowledge_graph_config
                print(f"Loaded knowledge graph config: {pipe.knowledge_graph_config.name}")
                break
        else:
            print("No kg_retrieval pipe found in the baseline pipeline config")

    def set_baseline_by_config(self, baseline: PipelineConfig):
        """
        Sets the baseline Pipeline Config for the experiment.

        Args:
            baseline (PipelineConfig): The baseline pipeline config.
        """
        self.baseline = baseline

    def set_baseline_by_path(self, baseline_path: str):
        """
        Sets the baseline Pipeline Config for the experiment by loading it from a file.

        Args:
            baseline_path (str): The path to the baseline pipeline config file.
        """
        if not self.fpm.file_path_exists(baseline_path):
            raise ValueError(f"File {baseline_path} does not exist")
        loaded_configs = self.load_configs_from_path(
            baseline_path, PipelineConfig)
        if len(loaded_configs) > 1:
            print("Found multiple pipeline configs, using the first one")
        self.baseline = loaded_configs[0]
        print(f"Loaded pipeline config: {self.baseline.name}")

    def add_parameter_range(self, parameter_range: ParameterRange):
        """
        Adds a parameter range to the experiment which is used for tuning a
        specific parameter of the baseline.

        Args:
            parameter_range (ParameterRange): The parameter range to add.
        """
        self.parameter_ranges.append(parameter_range)

    def load_parameter_ranges_from_path(self, path: str):
        """
        Loads parameter ranges from a JSON file.

        Args:
            path (str): The path to the JSON file containing parameter ranges.
        """
        if not self.fpm.file_path_exists(path):
            raise ValueError(f"File {path} does not exist")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for param_range in data:
            self.parameter_ranges.append(
                ParameterRange.model_validate(param_range))
        print(f"Loaded {len(data)} parameter ranges from {path}")

    def add_evaluator(self, evaluator: EvaluatorConfig):
        """
        Adds an evaluator to the experiment.

        Args:
            evaluator (EvaluatorConfig): The evaluator to add.
        """
        self.evaluators.append(evaluator)

    def load_evaluators_from_path(self, path: str):
        """
        Loads the evaluators from a JSON file.

        Args:
            path (str): The path to the JSON file containing evaluator configs.
        """
        if not self.fpm.file_path_exists(path):
            raise ValueError(f"File {path} does not exist")
        loaded_configs = self.load_configs_from_path(path, EvaluatorConfig)
        for evaluator in loaded_configs:
            self.evaluators.append(evaluator)
        print(f"Loaded {len(loaded_configs)} evaluator configs from {path}")

    def add_qa_dataset(self, dataset: DatasetConfig):
        """
        Sets the QA dataset for the experiment.

        Args:
            dataset (DatasetConfig): The QA dataset to add.
        """
        self.qa_dataset = dataset

    def load_qa_dataset_from_path(self, path: str):
        """
        Loads the configuration for the QA dataset from a JSON file.

        Args:
            path (str): The path to the JSON file containing the dataset config.
        """
        if not self.fpm.file_path_exists(path):
            raise ValueError(f"File {path} does not exist")
        loaded_configs = self.load_configs_from_path(path, DatasetConfig)
        if len(loaded_configs) > 1:
            print("Found multiple dataset configs, using the first one")
        self.qa_dataset = loaded_configs[0]
        print(f"Loaded dataset config: {self.qa_dataset.name}")

    def load_configs_from_path(self, 
                               path: str, 
                               config_type: Type[Config]) -> List[Config]:
        """
        Loads configurations from the specified path.

        Args:
            path (str): The file path to load configurations from.
            config_type (Type[Config]): The type of configuration to load.
        """
        config_manager = self.config_manager_factory.get_config_manager_by_type(
            config_type)
        try:
            config_manager.load_configs_from_path(
                file_path=path,
                overwrite_existing=True,
                throw_on_error=True
            )
        except Exception as e:
            print(
                f"Error loading {config_type.__name__} config from {path}: {e}")
            raise e
        loaded_configs = config_manager.get_all_configs()
        if len(loaded_configs) == 0:
            raise ValueError(
                f"No {config_type.__name__} config found in {path}")
        return loaded_configs
