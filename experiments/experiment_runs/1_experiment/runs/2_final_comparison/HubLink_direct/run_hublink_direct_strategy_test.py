import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from implementation.resource_management import FilePathManager
from implementation.experimentation.experiment_config_builder import ExperimentConfigBuilder
from implementation.experimentation.experiment_runner import ExperimentRunner, ExperimentRunnerSettings


def main():
    """Main function to run the experiment."""
    
    # -----------------------
    # -- Prepare the paths --
    # -----------------------
    fpm = FilePathManager()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    results_path = fpm.combine_paths(current_directory, "results")
    evaluator_configs_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 2),
        "evaluator_configs.json"
    )
    base_config_path = fpm.combine_paths(
        current_directory,
        "hublink_direct_strategy_config.json",
    )
    qa_dataset_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 2),
        "qa_datasets",
        "qa_datasets",
        "full",
        "deep_distributed_graph_dataset.csv"
    )

    # --------------------------------
    # -- Build the ExperimentConfig --
    # --------------------------------
    exp_config_builder = ExperimentConfigBuilder()
    exp_config_builder.set_baseline_by_path(base_config_path)
    exp_config_builder.load_evaluators_from_path(evaluator_configs_path)
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
            weave_project_name="kastel-sdq-meta-research/ma_mschneider_hublink_direct_strategy_test"
        )
    )

    results = runner.run()
    print(results.head())


if __name__ == '__main__':
    main()
