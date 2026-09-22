from typing import Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sqa_system.core.logging.logging import get_logger

from .base.base_plotter import BasePlotter

logger = get_logger(__name__)


class MetricBoxplotsPlotter(BasePlotter):
    """
    Create box plots for each metric, grouped by configuration. The x-axis
    represents the metric name, and the y-axis represents the metric values.
    """

    def plot(self, vis_folder: Optional[str]):
        if self.data is None or self.data.empty:
            logger.warning("No data available to plot boxplots.")
            return

        # Get summary data using the evaluators
        summary_data = self._calculate_summary_scores()
        if summary_data.empty:
            logger.warning("No summary data available for boxplots.")
            return

        plt.figure(figsize=(15, 10))
        sns.boxplot(
            data=summary_data,
            x="Metric",
            y="Average Score",
            hue="config_hash",
            palette=self.COLOR_LIST
        )
        plt.xticks(rotation=45)
        plt.title(f"Box Plots for {self.group_name} Metrics by Configuration")
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.legend(title="Configuration", bbox_to_anchor=(
            1.05, 1), loc="upper left")
        plt.tight_layout()

        if self.settings.should_save_to_file and vis_folder:
            os.makedirs(os.path.join(vis_folder, "metric_boxplots"), exist_ok=True)
            plt.savefig(os.path.join(vis_folder, 
                                     "metric_boxplots",
                                     f"{self.group_name}_metric_boxplots.{self.settings.file_type}"))

        if self.settings.should_print:
            plt.show()
        plt.close()
