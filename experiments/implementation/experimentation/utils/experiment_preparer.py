from copy import deepcopy
from typing import Any, Tuple, Callable, List, Optional, Union
from weave.flow.scorer import Scorer

from implementation.core import Config, DatasetConfig
from implementation.config.config_models.experiment_config import ExperimentConfig
from implementation.config.config_models.pipeline_config import PipelineConfig
from implementation.data_loader.dataset_manager import DatasetManager
from implementation.shared_models.qa_dataset import QADataset
from implementation.experimentation.evaluation.factory.evaluator_factory import EvaluatorFactory
from implementation.core.logging import get_logger
logger = get_logger(__name__)


class ExperimentPreparer:
    """
    Class responsible for the preparation of an experiment.

    Args:
        experiment_config (ExperimentConfig): The configuration of the experiment.
        qa_dataset_path (str): The path to the QA dataset that should be used to
            conduct the experiment. If not specified, the dataset from the
            experiment config will be used so this parameter is optional.
    """

    def __init__(self,
                 experiment_config: ExperimentConfig,
                 qa_dataset_path: Optional[str] = None) -> None:
        self.experiment_config = experiment_config
        self.qa_dataset_path = qa_dataset_path

    def prepare_dataset(self) -> List[dict]:
        """
        Loads the dataset specified in the experiment config and prepares it for the experiment.

        This step is required because weave expects the dataset to be a list of dictionaries.
        The score functions of the evaluators can than access the data by the keys of the
        dictionaries.

        Returns:
            List[dict]: The prepared dataset as a list of dictionaries.
        """
        if not self.qa_dataset_path and not self.experiment_config.qa_dataset:
            raise ValueError(
                "No QA dataset path or QA dataset config specified")

        dataset_config = self.experiment_config.qa_dataset
        if not dataset_config:
            dataset_config = DatasetConfig(
                file_name="",
                loader="CSVQALoader",
                additional_params={},
                loader_limit=-1
            )
        dataset = DatasetManager().get_dataset(
            config=dataset_config,
            file_path=self.qa_dataset_path)
        if not isinstance(dataset, QADataset):
            raise ValueError("Dataset is not of type QADataset")

        data = dataset.get_all_entries()

        prepared_dataset = []
        for entry in data:
            prepared_dataset.append(entry.model_dump())

        return prepared_dataset

    def prepare_evaluators(self) -> List[Union[Callable[..., Any], Scorer]]:
        """
        Prepares the evaluators specified in the experiment config.
        """
        evaluators = []
        for evaluator_config in self.experiment_config.evaluators:
            evaluators.append(EvaluatorFactory.create(evaluator_config))
        return evaluators

    def prepare_pipeline_configs(self, skip_base_config: bool = False) -> List[PipelineConfig]:
        """
        Generates configurations based on the one-factor-at-a-time method.
        It looks up the parameter ranges in the experiment config and generates
        configurations for each combination of parameter values.

        Args:
            skip_base_config (bool): If True, the base configuration will be skipped
                and only the generated configurations will be returned.

        Returns:
            List[PipelineConfig]: A list of generated configurations.
        """
        base_config = self.experiment_config.base_pipeline_config
        param_ranges = self.experiment_config.parameter_ranges

        if not param_ranges:
            return [base_config]

        config_hashes = set()
        config_hashes.add(base_config.config_hash)
        generated_configs = [base_config]

        for param_range in param_ranges:
            for value in param_range.values:
                config = deepcopy(base_config)
                all_configs = config.gather_all_configs()
                config_to_modify = all_configs[param_range.config_name]
                self._set_config_param(
                    config_to_modify,
                    (param_range.parameter_name, value),
                    list_index=param_range.list_index,
                    dict_key=param_range.dict_key)
                hash_value = config.config_hash
                if hash_value in config_hashes:
                    continue
                config_hashes.add(hash_value)
                generated_configs.append(config)

        if skip_base_config:
            generated_configs = generated_configs[1:]

        return generated_configs

    def _set_config_param(self,
                          config: Config,
                          param_name_value: Tuple[str, Any],
                          list_index: Optional[int] = None,
                          dict_key: Optional[str] = None):
        """
        This function is a helper that allows to set a parameter in the config object. It checks
        whether the parameter is a list or a dictionary and sets the value accordingly.

        Args:
            config (Config): The config object to set the parameter in.
            param_name_value (Tuple[str, Any]): The name and value of the parameter to set.
            list_index (Optional[int]): The index of the list element to set. If None, the
                parameter is set directly.
            dict_key (Optional[str]): The key of the dictionary element to set. If None,
                the parameter is set directly.
        Raises:
            AttributeError: If the parameter name is not found in the config object.
        """
        try:
            param_name, param_value = param_name_value
            # get the type of the parameter in the config
            param_type = type(getattr(config, param_name))

            # First we check whether we have a a list
            # and if the index is set. This indicates that
            # we are tuning a specific value inside of a list
            # which is the case for the pipe configs
            if issubclass(param_type, List) and list_index is not None:
                first_list_element = getattr(config, param_name)[list_index]
                if isinstance(first_list_element, Config):
                    new_list = getattr(config, param_name)
                    new_config = first_list_element.model_validate(param_value)
                    new_list[list_index] = new_config
                    setattr(config, param_name, new_list)
                else:
                    new_list = getattr(config, param_name)
                    new_list[list_index] = param_value
                    setattr(config, param_name, new_list)

            # Next we check whether we have a dictionary and if the
            # dict key is set. This indicates that we are tuning a
            # specific value inside of a dictionary
            elif issubclass(param_type, dict) and dict_key is not None:
                current_dict = getattr(config, param_name)
                if isinstance(dict_key, list):
                    if not isinstance(param_value, list) and len(param_value) == len(dict_key):
                        raise ValueError(
                            "If the dict key is a list, the parameter " +
                            "value must be a list of the same length")

                    for index, key in enumerate(dict_key):
                        current_dict[key] = param_value[index]
                else:
                    current_dict[dict_key] = param_value
                setattr(config, param_name, current_dict)

            # Here we check if the type of the parameter is a config
            # If that is the case, we recursively call this function
            elif issubclass(param_type, Config):
                new_config = param_type.model_validate(param_value)
                setattr(config, param_name, new_config)
            else:
                setattr(config, param_name, param_value)

        except AttributeError as exc:
            raise AttributeError(
                f"Config does not have attribute '{param_name}'") from exc
