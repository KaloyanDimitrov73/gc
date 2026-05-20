from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sqa_system.core.logging.logging import get_logger
from .base.base_plotter import BasePlotter

logger = get_logger(__name__)

MAX_BARS = 60


class MetricByColumnPerConfigPlotter(BasePlotter):
    """
    For a given column, this plotter creates one plot per available metric. 
    In each plot, the x-axis represents the unique values of the column, 
    the y-axis the average metric score, and the bar colors indicate the configuration hash.
    """

    def __init__(self, column_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_name = column_name

    def plot(self, vis_folder: Optional[str]):
        if self.data is None or self.data.empty:
            logger.warning("No data available to plot average metrics.")
            return

        if self.column_name not in self.data.columns:
            logger.warning(
                f"Grouping column '{self.column_name}' not found in data.")
            return

        summary_data = self._summarize_data()

        if summary_data.empty:
            logger.warning("No summary data available after calculation.")
            return

        # For each available metric we generate a separate plot
        for metric in summary_data["Metric"].unique():
            metric_df = summary_data[summary_data["Metric"] == metric]
            unique_values = sorted(metric_df[self.column_name].unique())
            num_configs = metric_df["config_hash"].nunique()
            num_values = len(unique_values)
            complexity = num_configs * num_values

            if complexity > MAX_BARS:
                max_values = max(1, int(np.ceil(MAX_BARS / num_configs)))
                for part, start in enumerate(range(0, num_values, max_values), start=1):
                    chunk_values = unique_values[start:start + max_values]
                    chunk_df = metric_df[metric_df[self.column_name].isin(chunk_values)]
                    self._plot_internal(chunk_df, vis_folder, metric, part)
            else:
                self._plot_internal(metric_df, vis_folder, metric)
                
    def _summarize_data(self):
        """
        Helper function to summarize the data by calculating average scores
        for each unique value in the specified column.
        """
        summary_data = pd.DataFrame()
        for column_value in self.data[self.column_name].unique():
            filtered_data = self.data[self.data[self.column_name] == column_value].copy()
            column_summary = self._calculate_summary_scores(data=filtered_data)
            column_summary[self.column_name] = column_value
            summary_data = pd.concat([summary_data, column_summary])
        return summary_data

    def _plot_internal(
        self,
        df: pd.DataFrame,
        vis_folder: Optional[str],
        metric: str,
        part: int = None
    ):
        """
        Helper function to plot the average metric scores by column values.
        
        Args:
            df: DataFrame containing the average scores per configuration.
            vis_folder: The folder where the plot should be saved.
            metric: The name of the metric to plot.
            part: The part number for splitting the metrics if needed.
        """
        plt.figure(figsize=(15, 10))

        unique_configs = df["config_hash"].unique()
        palette = {
            cfg: "#e32619" if cfg == self.settings.baseline_config else self.COLOR_LIST[i % len(
                self.COLOR_LIST)]
            for i, cfg in enumerate(unique_configs)
        }

        sns.barplot(
            x=self.column_name,
            y="Average Score",
            hue="config_hash",
            data=df,
            palette=palette
        )

        plt.title(
            f"Average of the '{metric}' metric by '{self.column_name}'")
        plt.xlabel(self.column_name)
        plt.ylabel("Average Score")
        plt.xticks(rotation=90)
        plt.legend(title="Configuration", bbox_to_anchor=(
            1.05, 1), loc="upper left")
        plt.tight_layout()

        if self.settings.should_save_to_file and vis_folder:
            base = self.file_path_manager.combine_paths(
                vis_folder,
                f"average_metric_by_{self.column_name}",
                f"{metric}_average_{self.group_name}_by_{self.column_name}.{self.settings.file_type}"
            )
            if part is not None:
                base = base.replace(
                    f".{self.settings.file_type}", f"_part_{part}.{self.settings.file_type}"
                )
            self.file_path_manager.ensure_dir_exists(base)
            plt.savefig(base)

        if self.settings.should_print:
            plt.show()
        plt.close()
