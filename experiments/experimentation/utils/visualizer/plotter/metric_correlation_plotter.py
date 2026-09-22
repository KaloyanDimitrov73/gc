from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sqa_system.core.logging.logging import get_logger

from .base.base_plotter import BasePlotter

logger = get_logger(__name__)


class MetricCorrelationPlotter(BasePlotter):
    """
    Plot a heatmap to show the correlation between different metrics.
    """

    def plot(self, vis_folder: Optional[str]):
        if self.data is None or self.data.empty:
            logger.warning("No data available for correlation plots.")
            return
        
        summary_data = self._calculate_summary_scores()
        if summary_data.empty:
            logger.warning("No summary data available for correlation plot.")
            return

        # Pivot the data to create a correlation matrix
        pivot_data = summary_data.pivot(
            index="config_hash", 
            columns="Metric", 
            values="Average Score"
        ).reset_index()

        if pivot_data.shape[1] < 3:
            logger.warning("Not enough metrics available for correlation plot.")
            return

        corr = pivot_data.drop(columns=["config_hash"]).corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation Heatmap of {self.group_name} Metrics")
        plt.tight_layout()

        if self.settings.should_save_to_file and vis_folder:
            file_path = self.file_path_manager.combine_paths(
                vis_folder, 
                "metric_correlation",
                f"{self.group_name}_metric_correlation.{self.settings.file_type}"
            )
            self.file_path_manager.ensure_dir_exists(file_path)
            plt.savefig(file_path)

        if self.settings.should_print:
            plt.show()
        plt.close()
