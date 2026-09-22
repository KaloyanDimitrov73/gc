import sys, os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import os
from experimentation.utils.visualizer.experiment_visualizer import (
    ExperimentVisualizer, ExperimentVisualizerSettings, PlotType)


def main():
    """
    Main function to visualize the results of the experiments in the subfolders.
    """
    current_directory = os.path.dirname(os.path.realpath(__file__))

    print("Visualizing the results of the experiments in the subfolders...")

    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=current_directory,
            should_save_to_file=True,
            save_folder_path=os.path.join(current_directory, "visualization"),
            should_print=False,
            config_to_name_mapping={
                "8f2d9a1435d71747bb0b31ba7527678a": "mxbai-embed-large",
                "677b41d533cf570efb4edbc45d94e216": "granite-embedding",
                "b310a26a02daf3f1e8543d7fa27723c9": "bge-m3",
                "bc2992ac3dbc124c73bb01e52b355103": "text-embedding-3-large",
                "d36fe6ad2bf8dfbdabc02132b0b90802": "text-embedding-3-small",
                "f061d8025edb8ee3769efd457f035e3c": "Alibaba-NLP/gte-Qwen2-7B-instruct",
            },
            file_type="png"
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_PER_CONFIG,
            PlotType.TABLE
        ]
    )
    print("Finished visualizing the results of the experiments in the subfolders.")


if __name__ == '__main__':
    main()
