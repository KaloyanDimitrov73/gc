from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

from sqa_system.core.logging.logging import get_logger

from .base.base_plotter import BasePlotter

logger = get_logger(__name__)

class MetricDistributionsPlotter(BasePlotter):
    """
    Generate KDE plots/histograms for each metric and configuration.
    Creates separate plots for each configuration.
    """

    def plot(self, vis_folder: Optional[str]):
        if self.data is None or self.data.empty:
            logger.warning("No data available to plot distributions.")
            return

        summary_data = self._calculate_summary_scores()
        if summary_data.empty:
            logger.warning("No summary data available for distribution plots.")
            return

        configs = summary_data["config_hash"].unique()
        for config in configs:
            config_data = summary_data[summary_data["config_hash"] == config]
            if config_data.empty:
                continue

            metrics = config_data["Metric"].unique()
            n_metrics = len(metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols if n_metrics > 0 else 1

            plt.figure(figsize=(20, 5 * n_rows))
            for i, metric in enumerate(metrics):
                metric_data = config_data[config_data["Metric"] == metric]
                plt.subplot(n_rows, n_cols, i + 1)
                sns.histplot(data=metric_data, x="Average Score", kde=True)
                plt.title(f"Distribution of {metric}\nConfig: {config}")
                plt.xlabel(metric)
                plt.ylabel("Frequency")
            plt.tight_layout()

            if self.settings.should_save_to_file and vis_folder:
                if self.group_name:
                    file_path = self.file_path_manager.combine_paths(
                        vis_folder, 
                        "metric_distributions",
                        f"{self.group_name}_config_{config}.{self.settings.file_type}"
                    )
                    
                else:
                    file_path = self.file_path_manager.combine_paths(
                        vis_folder, 
                        "metric_distributions",
                        f"config_{config}.{self.settings.file_type}"
                    )
                self.file_path_manager.ensure_dir_exists(file_path)
                plt.savefig(file_path)

            if self.settings.should_print:
                plt.show()
            plt.close()