import sys, os as _os; sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../../../../..'))
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
                "f6fa46e12a5ab5344133f23580a031c8": "DiFaR",
                "a389979bfc391531ba1b899efc244a87": "FiDeLiS 1",
                "ba4a27f219fc1d6411b51229990eb688": "FiDeLiS 2",
                "994c7d647defa7c77253442203e44879": "HubLink 1",
                "08363ed1f80d492dc324a1294c21c90b": "HubLink 2",
                "ce2030671a6ea6d938106e416142cf8f": "Mindmap",
                "a389979bfc391531ba1b899efc244a87": "StructGpt 1",
                "ba4a27f219fc1d6411b51229990eb688": "StructGpt 2",
                "2b63b4c9b3804801c14d7a2d8a2fb350": "ToG 1",
                "a7637f8ac7438caaa5b1a239d294ef0f": "ToG 2",
            }
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
