import os

from implementation.file_path_management import FilePathManager
from implementation.experimentation.experiment_config_builder import ExperimentConfigBuilder
from implementation.experimentation.experiment_runner import ExperimentRunner, ExperimentRunnerSettings
from implementation.experimentation.utils.experiment_runner_settings import ExecutionStrategyType


def main():
    """Main function to run the experiment."""
    # -----------------------
    # -- Prepare the paths --
    # -----------------------
    fpm = FilePathManager()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    folder_path = fpm.combine_paths(current_directory, "results")
    qa_dataset_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 1),
        "test_qa_dataset.csv"
    )
    evaluator_configs_path = fpm.combine_paths(
        current_directory,
        "evaluator_configs.json"
    )
    baseline_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 3),
        "base_configs",
        "base_config_kit.json"
    )

    # --------------------------------
    # -- Build the ExperimentConfig --
    # --------------------------------
    exp_config_builder = ExperimentConfigBuilder()
    exp_config_builder.set_baseline_by_path(baseline_path)
    exp_config_builder.load_evaluators_from_path(evaluator_configs_path)
    experiment_config = exp_config_builder.build()

    # ------------------------
    # -- Run the Experiment --
    # ------------------------
    runner = ExperimentRunner(
        experiment_config=experiment_config,
        settings=ExperimentRunnerSettings(
            results_folder_path=fpm.combine_paths(
                folder_path, "test_run", experiment_config.config_hash),
            qa_data_path=qa_dataset_path,
            debugging=True,
            log_to_results_folder=True,
            # Add name of weave project here
            weave_project_name="experiment_tests",
            number_of_workers=3,
            number_of_processes=1,
            execution_strategy=ExecutionStrategyType.SEQUENTIAL,
            skip_weave=True
        ),
    )

    results = runner.run()
    print(results.head())


if __name__ == '__main__':
    main()
