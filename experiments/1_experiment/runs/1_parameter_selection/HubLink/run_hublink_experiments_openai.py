import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from sqa_system.core.data.file_path_manager import FilePathManager
from experimentation.experiment_config_builder import ExperimentConfigBuilder
from experimentation.experiment_runner import ExperimentRunner, ExperimentRunnerSettings


def main():
    """Main function to run the experiment."""
    # -----------------------
    # -- Prepare the paths --
    # -----------------------
    fpm = FilePathManager()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    results_path = fpm.combine_paths(current_directory, "results_openai")
    qa_dataset_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 4),
        "qa_datasets",
        "qa_datasets",
        "reduced",
        "reduced_deep_distributed_graph_dataset.csv"
    )
    evaluator_configs_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 4),
        "evaluator_configs.json"
    )
    tuning_params_path = fpm.combine_paths(
        current_directory,
        "tuning_parameters_openai.json"
    )
    baseline_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 3),
        "base_configs",
        "HubLink",
        "base_config_openai.json"
    )

    # --------------------------------
    # -- Build the ExperimentConfig --
    # --------------------------------
    exp_config_builder = ExperimentConfigBuilder()
    exp_config_builder.set_baseline_by_path(baseline_path)
    exp_config_builder.load_evaluators_from_path(evaluator_configs_path)
    exp_config_builder.load_parameter_ranges_from_path(tuning_params_path)
    experiment_config = exp_config_builder.build()

    # ------------------------
    # -- Run the Experiment --
    # ------------------------
    runner = ExperimentRunner(
        experiment_config=experiment_config,
        settings=ExperimentRunnerSettings(
            results_folder_path=fpm.combine_paths(
                results_path, experiment_config.config_hash),
            qa_data_path=qa_dataset_path,
            debugging=True,
            log_to_results_folder=True,
            # Add name of weave project here
            # weave_project_name="Todo",
            number_of_workers=10,
            number_of_processes=1,
            skip_base_config= True
        ),
    )

    results = runner.run()
    print(results.head())


if __name__ == '__main__':
    main()
