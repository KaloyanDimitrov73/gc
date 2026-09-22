from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sqa_system.core.logging.logging import get_logger

from .base.base_plotter import BasePlotter

logger = get_logger(__name__)

MAX_BARS = 60


class AverageMetricsPerConfigPlotter(BasePlotter):
    """
    Plot the average of each metric per configuration. This plotter has
    on the x-axis the metric name and on the y-axis the average score.
    The bars are colored based on the configuration hash.
    """

    def plot(self, vis_folder: Optional[str]):
        if self.data is None or self.data.empty:
            logger.warning("No data available to plot average metrics.")
            return
        
        summary_data = self._calculate_summary_scores()

        # Calculate complexity
        config_list = summary_data["config_hash"].unique()
        num_configs = len(config_list)
        metric_list = summary_data["Metric"].unique()
        num_metrics = len(metric_list)
        complexity = num_configs * num_metrics
        summary_data = summary_data.sort_values(by="config_hash")

        if complexity > MAX_BARS:
            max_metrics = max(1, int(np.ceil((MAX_BARS / num_configs))))
            for part, i in enumerate(range(0, num_metrics, max_metrics), start=1):
                group_metrics = metric_list[i:i + max_metrics]
                melted_group = summary_data[summary_data["Metric"].isin(
                    group_metrics)]
                self._plot_interal(melted_group, vis_folder, part=part)
        else:
            self._plot_interal(summary_data, vis_folder)

    def _plot_interal(self,
                      mean_scores: pd.DataFrame,
                      vis_folder: Optional[str],
                      part: int = None):
        """
        Helper function to plot the average metrics per configuration.
        
        Args:
            mean_scores: DataFrame containing the average scores per configuration.
            vis_folder: The folder where the plot should be saved.
            part: The part number for splitting the metrics if needed.
        """

        plt.figure(figsize=(15, 10))

        unique_configs = mean_scores["config_hash"].unique()
        palette = {
            cfg: "#e32619" if cfg == self.settings.baseline_config else self.COLOR_LIST[i % len(
                self.COLOR_LIST)]
            for i, cfg in enumerate(unique_configs)
        }
        sns.barplot(
            x="Metric",
            y="Average Score",
            hue="config_hash",
            data=mean_scores,
            palette=palette
        )

        plt.legend(title="Configuration", bbox_to_anchor=(
            1.05, 1), loc="upper left")

        plt.title(f"Average {self.group_name} Metrics per Configuration")
        plt.xlabel("Evaluation Metric")
        plt.ylabel("Average Score")
        plt.legend(title="Configuration", bbox_to_anchor=(
            1.05, 1), loc="upper left")
        plt.xticks(rotation=90)
        plt.tight_layout()

        if self.settings.should_save_to_file and vis_folder:
            if self.group_name:
                folder = self.file_path_manager.combine_paths(
                    vis_folder,
                    "average_metrics_per_config",
                    f"average_{self.group_name}_metrics_per_config.{self.settings.file_type}")

            else:
                folder = self.file_path_manager.combine_paths(
                    vis_folder,
                    "average_metrics_per_config",
                    f"average_metrics_per_config.{self.settings.file_type}")

            if part is not None:
                folder = folder.replace(
                    f".{self.settings.file_type}", f"_part_{part}.{self.settings.file_type}")

            self.file_path_manager.ensure_dir_exists(folder)
            plt.savefig(folder)

        if self.settings.should_print:
            plt.show()
        plt.close()
