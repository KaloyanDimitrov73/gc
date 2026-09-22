from typing import Optional
import pandas as pd

from sqa_system.core.logging.logging import get_logger

from .base.base_plotter import BasePlotter

logger = get_logger(__name__)


class MetricsTablePlotter(BasePlotter):
    """
    Generate a table showing the average score of each metric for each configuration.
    """

    def plot(self, vis_folder: Optional[str]):
        if self.data is None or self.data.empty:
            logger.warning("No data available to plot metrics table.")
            return

        summary_data = self._calculate_summary_scores()
        if summary_data.empty:
            logger.warning("No summary data available for metrics table.")
            return

        stats_df = summary_data.pivot(
            index="config_hash",
            columns="Metric",
            values="Average Score"
        ).reset_index()

        # Get baseline metrics if a baseline config is specified
        base_config = self.settings.baseline_config
        if base_config and base_config in stats_df["config_hash"].values:
            base_df = stats_df[stats_df["config_hash"] == base_config]
            metrics = stats_df.columns[1:]  # Skip config_hash column

            formatted_df = pd.DataFrame()
            formatted_df["Config"] = stats_df["config_hash"]

            # For each metric, format the values with percentage diff from baseline
            for metric in metrics:
                base_value = base_df[metric].values[0]

                formatted_df[metric] = stats_df.apply(self.format_with_diff,
                                                      axis=1,
                                                      metric=metric,
                                                      base_value=base_value)

            stats_df = formatted_df
        else:
            # No baseline, just format the values
            stats_df = stats_df.rename(columns={"config_hash": "Config"})
            for col in stats_df.columns[1:]:
                stats_df[col] = stats_df[col].apply(
                    lambda x: f"{x:.3f}" if not pd.isna(x) else "none")

        if self.settings.should_save_to_file and vis_folder:
            file_path = self.file_path_manager.combine_paths(
                vis_folder,
                "metrics_table",
                f"{self.group_name}_metrics_table.csv" if self.group_name else "metrics_table.csv"
            )
            self.file_path_manager.ensure_dir_exists(file_path)
            stats_df.to_csv(file_path, index=False)

        if self.settings.should_print:
            print(stats_df)

    def format_with_diff(self, row, metric, base_value):
        """
        Format the metric value with percentage difference from the baseline.
        """
        value = row[metric]
        if row["config_hash"] == self.settings.baseline_config or pd.isna(value) or pd.isna(base_value):
            return f"{value:.3f}" if not pd.isna(value) else "N/A"

        pct_diff = ((value - base_value) / base_value) * 100
        return f"{value:.3f} ({pct_diff:+.1f}%)"
