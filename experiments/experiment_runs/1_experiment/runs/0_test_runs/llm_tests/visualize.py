import sys, os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import os
from implementation.experimentation.utils.visualizer.experiment_visualizer import (
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
                "8f2d9a1435d71747bb0b31ba7527678a": "qwen2.5:14b",
                "1d53f76bca9f2c8b3fe9520a54fffe2c": "deepseek-r1:8b",
                "8bc27105c0bf9b3bb283178fe8e03336": "gpt-4o",
                "36d93fec626fe6870c0f6cb16c4be800": "llama3.1",
                "e43a9aca281bf38b1424265df946cf48": "gpt-4o-mini"
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
