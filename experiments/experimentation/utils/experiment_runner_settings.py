from enum import Enum
from dataclasses import dataclass


class ExecutionStrategyType(str, Enum):
    """The type of the experiment execution"""
    SEQUENTIAL = "sequential"
    PARALLEL_EVALUATION = "parallel_evaluation"
    NO_EVALUATION = "no_evaluation"


@dataclass
class ExperimentRunnerSettings:
    """
    Dataclass for ExperimentRunner Settings

    Args:
        results_folder_path (str): The path to the folder where the results 
            should be saved.
        qa_data_path (str): The path where the QA Data should be loaded from
        weave_project_name (str): The name of the Weave project where the 
            experiment will be logged.
        debugging (bool): Whether to enable debugging mode.
        log_to_results_folder (bool): Whether to log to the results folder.
        execution_strategy (ExecutionStrategyType): The execution strategy to use.
        number_of_workers (int): The number of workers that are used if the 
            strategy allows parallel execution.
        number_of_processes (int): The number of processes to distribute the 
            generated configurations to.
        visualize_results (bool): Whether to visualize the results in diagrams
            using the experiment visualizer.
        skip_base_config (bool): Whether to run the base config or not.
    """
    results_folder_path: str = None
    qa_data_path: str = None
    weave_project_name: str = None
    debugging: bool = False
    log_to_results_folder: bool = False
    execution_strategy: ExecutionStrategyType = ExecutionStrategyType.SEQUENTIAL
    number_of_workers: int = 1
    number_of_processes: int = 1
    visualize_results: bool = False
    skip_base_config: bool = False
