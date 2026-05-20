from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sqa_system.core.logging.logging import get_logger
from .base.base_plotter import BasePlotter

logger = get_logger(__name__)

MAX_BARS = 60


class AverageMetricsByColumnPlotter(BasePlotter):
    """
    This plotter creates a bar plot for each configuration. It shows on the x-axis
    the unique values of a specified column, and on the y-axis the average score
    of each metric. The bars are colored based on the configuration hash.
    """

    def __init__(self, column_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_name = column_name

    def plot(self, vis_folder: Optional[str]):
        self._validate_data()

        # Group the data by the column before calculating summary scores
        if self.column_name not in self.data.columns:
            logger.warning(
                f"Grouping column '{self.column_name}' not found in data.")
            return

        summary_data = pd.DataFrame()
        for column_value in self.data[self.column_name].unique():
            column_data = self.data[self.data[self.column_name]
                                    == column_value].copy()
            column_summary = self._calculate_summary_scores(data=column_data)
            column_summary[self.column_name] = column_value
            summary_data = pd.concat([summary_data, column_summary])

        if summary_data.empty:
            logger.warning("No summary data available after calculation.")
            return

        self._run_generation(vis_folder, summary_data)

    def _run_generation(self,
                        vis_folder: Optional[str],
                        summary_data: pd.DataFrame):
        """
        For each unique configuration, generate a bar plot showing the average
        scores of the metrics grouped by the specified column.
        If the number of bars exceeds MAX_BARS, split the metrics into parts
        and generate separate plots for each part.

        Args:
            vis_folder: The folder where the plots should be saved.
            summary_data: DataFrame containing the summary data to plot.
        """
        for config in summary_data["config_hash"].unique():
            config_df = summary_data[summary_data["config_hash"] == config]
            num_groups = config_df[self.column_name].nunique()
            num_metrics = len(config_df["Metric"].unique())
            complexity = num_groups * num_metrics

            if complexity > MAX_BARS:
                max_metrics = max(1, int(np.ceil(MAX_BARS / num_groups)))
                metrics = config_df["Metric"].unique()
                for part, start in enumerate(range(0, len(metrics), max_metrics), start=1):
                    chunk = metrics[start:start + max_metrics]
                    sub_df = config_df[config_df["Metric"].isin(chunk)]
                    self._plot_internal(sub_df, vis_folder, config, part)
            else:
                self._plot_internal(config_df, vis_folder, config)

    def _validate_data(self):
        if self.data is None or self.data.empty:
            logger.warning("No data available to plot average metrics.")
            return

        if self.column_name not in self.data.columns:
            logger.warning(
                f"Grouping column '{self.column_name}' not found in data.")
            return

    def _plot_internal(
        self,
        df: pd.DataFrame,
        vis_folder: Optional[str],
        config: str,
        part: int = None
    ):
        """
        Helper function to create the bar plot for the given DataFrame.

        Args:
            df: DataFrame containing the data to plot.
            vis_folder: The folder where the plot should be saved.
            config: The configuration hash for the current plot.
            part: The part number for the current plot (if applicable).
        """
        df = df.groupby(
            ["Metric", self.column_name, "config_hash"],
            as_index=False
        )["Average Score"].mean()

        
        plt.figure(figsize=(15, 10))

        unique_values = df[self.column_name].unique()
        palette = {
            val: self.COLOR_LIST[i % len(self.COLOR_LIST)]
            for i, val in enumerate(unique_values)
        }

        sns.barplot(
            x="Metric",
            y="Average Score",
            hue=self.column_name,
            data=df,
            palette=palette
        )

        plt.title(
            f"Average {self.group_name} Metrics for config '{config}' grouped by '{self.column_name}'")
        plt.xlabel("Evaluation Metric")
        plt.ylabel("Average Score")
        plt.xticks(rotation=90)
        plt.legend(title=self.column_name,
                   bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if self.settings.should_save_to_file and vis_folder:
            base = self.file_path_manager.combine_paths(
                vis_folder,
                f"average_metrics_by_{self.column_name}",
                f"{config}_average_{self.group_name}_by_{self.column_name}.{self.settings.file_type}"
            )
            if part is not None:
                base = base.replace(
                    f".{self.settings.file_type}", f"_part_{part}.{self.settings.file_type}")
            self.file_path_manager.ensure_dir_exists(base)
            plt.savefig(base)

        if self.settings.should_print:
            plt.show()
        plt.close()
