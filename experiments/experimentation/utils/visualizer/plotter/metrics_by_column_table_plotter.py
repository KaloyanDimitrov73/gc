import os
from typing import Optional
import pandas as pd

from sqa_system.core.logging.logging import get_logger
from .base.base_plotter import BasePlotter

logger = get_logger(__name__)


class MetricsByColumnTablePlotter(BasePlotter):
    """
    This plotter generates a table for each configuration, showing the mean
    of each metric grouped by a specified column.
    """

    def __init__(self, column_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_name = column_name

    def plot(self, vis_folder: Optional[str]):
        if self.data is None or self.data.empty:
            logger.warning("No data available to plot metrics table.")
            return

        if self.column_name not in self.data.columns:
            logger.warning(f"Column '{self.column_name}' not found in data.")
            return

        filtered_data = self._filter_by_configs(self.data)
        configs = filtered_data["config_hash"].unique()

        for config in configs:
            self._plot_single_config(
                config=config,
                filtered_data=filtered_data,
                vis_folder=vis_folder
            )

    def _plot_single_config(self,
                            config: str,
                            filtered_data: pd.DataFrame,
                            vis_folder: Optional[str]):
        """
        Generates a table for a single configuration, showing the mean 
        of each metric grouped by the specified column.

        Args:
            config: The configuration hash for which to generate the table.
            filtered_data: The filtered DataFrame containing the data.
            vis_folder: The folder where the table should be saved.
        """
        config_data = filtered_data[filtered_data["config_hash"] == config]
        unique_values = config_data[self.column_name].unique()

        # Create a table for each unique value in the column
        stats_data = []
        for value in unique_values:
            value_data = config_data[config_data[self.column_name] == value].copy(
            )
            summary_data = self._calculate_summary_scores(data=value_data)

            if summary_data.empty:
                continue

            row_data = {self.column_name: value}
            # Extract metrics from summary data
            for metric in summary_data["Metric"].unique():
                metric_value = summary_data[summary_data["Metric"]
                                            == metric]["Average Score"].values
                if len(metric_value) > 0:
                    row_data[metric] = f"{metric_value[0]:.3f}"

            stats_data.append(row_data)

        if not stats_data:
            logger.warning(f"No summary data generated for config {config}")
            return

        stats_df = pd.DataFrame(stats_data)

        if self.settings.should_save_to_file and vis_folder:
            table_dir = self.file_path_manager.combine_paths(
                vis_folder,
                f"metrics_tables_by_{self.column_name}"
            )
            self.file_path_manager.ensure_dir_exists(table_dir)

            stats_df.to_csv(
                os.path.join(table_dir,
                             f"{config}_metrics_by_{self.column_name}.csv"),
                index=False)

        if self.settings.should_print:
            print(
                f"\nMetrics for config '{config}' grouped by '{self.column_name}':")
            print(stats_df)
            print("\n")
