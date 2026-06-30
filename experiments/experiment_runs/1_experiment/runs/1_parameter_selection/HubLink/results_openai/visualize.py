import sys, os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../../..'))
# A simple script to generate visualizations for the results of the
# Experiments.

# ----------------
# -- HOW TO USE --
# ----------------
# 1. Place this script in the same directory as the results of the
#    experiment(s) you want to visualize. It will automatically retrieve
#    all prediction files from the subfolders.
# 2. Run the script.
# (3.) Optionally you can change the variables inside of the script to change
#    what is plotted.

import os
from implementation.resource_management import FilePathManager
from implementation.experimentation.utils.visualizer.experiment_visualizer import (
    ExperimentVisualizer, ExperimentVisualizerSettings, PlotType)

# Here you replace the name of the configs with a unique
# name. Place the id on the left, and the name on the right.
CONFIG_TO_NAME_MAPPING: dict = {
    "8e6f3ff0a5eac369cfacbcc564f40bfb": "Base Config",
    "9942c5c0516c5e669f254c8254898e0a": "question-augmentation",
    "d057b51b9340c194c5a283cf143a55e2": "reranking",
    "a895306d6d76c1b631160b615fd620ec": "gpt-4o",
    "ce70df19126132bba7600d3709305425": "traversal-strategy",
    "d25f962dd188aad72919a1e8153ed580": "o3-mini",
    "1e97b6fa94c2a200d2bf43e314ac0460": "top-paths-to-keep: 20",
    "026e0e1798be6628a69b17f2c848e32e": "top-paths-to-keep: 30",
    "27d847ccaa42095750a184ad0d8272d1": "number-of-hubs: 20",
    "04373ebe014957aa44946cf803009034": "number-of-hubs: 30",
    "0cd074b1efc6462788a60172c257ccda": "path-weight-alpha: 9",
    "4e318228a92cafbf687f7fde13d584ab": "path-weight-alpha: 3",
    "8c954ee5c0c6e67b2fae962b1e104c22": "diversity-ranking: 0.01",
    "18244cdec7744695ab0dac91280fe424": "diversity-ranking: 0.0",
    "ba91a735ebb30fe3508658fbd1c9b971": "path-weight-alpha: 6",
    "be675783e128c37d531c64adeae37186": "diversity-ranking: 0.1",
    "f4b7f152c0c299e87c51a16bfbcf4054": "path-weight-alpha: 0",
    "7dcca2dae43942033f8722d09f09aa40": "no-output-filtering",
    "b473c12964d24c85c724a69a70876002": "no-question-components"
}
# The base config (or name if you replace it above)
# is highlighted in red in some plots. Should be a string.
BASELINE_CONFIG: str = "Base Config"


FPM = FilePathManager()
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
VISUALIZATIONS_DIR = FPM.combine_paths(
    CURRENT_DIRECTORY, "result_visualizations")
QA_DATASET_PATH = FPM.combine_paths(
    FPM.get_parent_directory(CURRENT_DIRECTORY, 5),
    "qa_datasets",
    "qa_datasets",
    "reduced",
    "reduced_deep_distributed_graph_dataset.csv"
)


def plot_average_metrics_per_config():
    """
    Generates a plot with the y-axis being the average of the metrics,
    the x-axis being the names of the metrics and the hue being the
    configuration hashes.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            save_folder_path=VISUALIZATIONS_DIR,
            qa_file_path=QA_DATASET_PATH,
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_PER_CONFIG,
            PlotType.TABLE
        ]
    )


def plot_average_metrics_by_column():
    """
    This function generates a plot with the y-axis beeing the average of the 
    metrics, the x-axis being the names of the metrics and the hue being the
    unique values of the column to group by.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            save_folder_path=VISUALIZATIONS_DIR,
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            qa_file_path=QA_DATASET_PATH,
            # It is recommended to add the config to visualize here else it will
            # generate many plots for each config that it finds.
            configs_to_visualize=[]
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_BY_COLUMN,
        ],
        column_to_group_by="retrieval_operation"
    )


def plot_metric_by_column():
    """
    This function generates per metric a plot with the metrics average value
    being on the y-axis, the unique values of the column on the x-axis and
    the hue being the configuration hashes.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            file_type="pdf",
            save_folder_path=VISUALIZATIONS_DIR,
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            qa_file_path=QA_DATASET_PATH,
            # Here you need to define the names of the metrics you want to
            # visualize. It has to be inside of a dictionary.
            metrics={
                "retrieval_operation": [
                    "recall_triples",
                ]
            }
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_PER_CONFIG
        ],
        column_to_group_by="retrieval_operation"
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_PER_CONFIG
        ],
        column_to_group_by="retrieval_operation"
    )
    
def plot_for_wiki():
    """
    Creates the plots used on the wiki page.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            file_type="png",
            save_folder_path=VISUALIZATIONS_DIR + "/wiki",
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            qa_file_path=QA_DATASET_PATH,
            # Here you need to define the names of the metrics you want to
            # visualize. It has to be inside of a dictionary.
            metrics={
                "": [
                    "hit@10_triples",
                    "map@10_triples",
                    "mrr@10_triples",
                    "f2_triples",
                    "precision_triples",
                    "recall_triples",
                    "bleu",
                    "rouge1_recall",
                    "bertscore"
                ]
            }
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_PER_CONFIG,
            PlotType.TABLE
        ]
    )


if __name__ == '__main__':
    plot_average_metrics_per_config()
    plot_metric_by_column()
    # plot_for_wiki()
