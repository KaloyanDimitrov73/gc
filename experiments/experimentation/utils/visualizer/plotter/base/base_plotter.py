from typing import List, Optional
from abc import ABC, abstractmethod
import pandas as pd
from pydantic import BaseModel, ConfigDict
import seaborn as sns

from sqa_system.core.data.file_path_manager import FilePathManager

sns.set_theme(style="whitegrid")


class PlotterSetings(BaseModel):
    """
    A class to hold the settings for a plotter.

    Args:
        configs_to_visualize: Optional subset of configurations to visualize.
        should_save_to_file: Whether or not to save the plot.
        file_type: The file type to save the plot as (e.g., 'pdf', 'png').
        should_print: Whether or not to show the plot in an interactive window.
        baseline_config: Optional baseline configuration for comparison.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    configs_to_visualize: Optional[List[str]]
    should_save_to_file: bool = False
    file_type: Optional[str] = "pdf"
    should_print: bool
    baseline_config: Optional[str] = None


class BasePlotter(ABC):
    """
    A base class for all plotters.
    This defines the common interface a plotter should implement.

    Args:
        data: DataFrame containing the data to plot.
        base_metrics: Dictionary of potential metrics to plot.
        settings: Settings for the plotter.
        group_name: Optional name to add a grouping name to the header
            of the plot.
    """
    COLOR_LIST = ["#3A6B35", "#E3B448", "#79ab74",
                  "#db8b30", "#B85042", "#F1AC88",
                  "#7560bf", "#6883BC", "#8A307F",
                  "#2F3C7E", "#FF69B4"]

    def __init__(self,
                 data: pd.DataFrame,
                 base_metrics: List[str],
                 settings: PlotterSetings,
                 group_name: str = None):
        self.data = data
        self.base_metrics = base_metrics
        self.settings = settings
        self.group_name = group_name
        self.file_path_manager = FilePathManager()

    @abstractmethod
    def plot(self, vis_folder: Optional[str]):
        """
        This method should be overridden by subclasses
        to implement the actual plotting logic.

        Args:
            vis_folder: The folder where the plot should be saved.
        """
        raise NotImplementedError("Subclasses must override this method.")

    def _calculate_summary_scores(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate summary scores for the metrics in the DataFrame.

        Args:
            data: Optional DataFrame to calculate scores from. If None, uses self.data.

        Returns:
            DataFrame with summary scores in a melted format with columns:
            "config_hash", "Metric", "Average Score"
        """
        df = data if data is not None else self.data

        if df is None or df.empty:
            return pd.DataFrame()

        filtered_data = self._filter_by_configs(df)
        rows: list[dict] = []
        for config in filtered_data["config_hash"].unique():
            config_data = filtered_data[filtered_data["config_hash"] == config]
            if config_data.empty:
                continue

            summary_row: dict[str, float] = {}
            for metric in self.base_metrics:
                for col in config_data.columns:
                    if metric in col:
                        summary_row[col] = config_data[col].mean()

            summary_row["config_hash"] = config
            rows.append(summary_row)

        df = pd.DataFrame(rows)
        return df.melt(
            id_vars=["config_hash"],
            var_name="Metric",
            value_name="Average Score",
        )

    def _get_available_metrics(self) -> List[str]:
        """
        Checks the base_metrics and returns a list of available metrics that are actually
        present in the DataFrame columns.

        Returns:
            List[str]: A list of available metrics that are present in the DataFrame columns.
        """
        if self.base_metrics is None:
            return []

        metrics_to_visualize = []
        if isinstance(self.base_metrics, dict):
            # If base_metrics is a dictionary, use its values
            metrics_to_visualize = list(self.base_metrics.values())
            # merge the lists
            metrics_to_visualize = [
                item for sublist in metrics_to_visualize for item in sublist]
        elif isinstance(self.base_metrics, list) and self.base_metrics and isinstance(self.base_metrics[0], dict):
            # Handle list of dictionaries
            metrics_to_visualize = [
                item for metric in self.base_metrics for item in metric.values()]
        else:
            metrics_to_visualize = self.base_metrics
        available_metrics = []
        for metric in metrics_to_visualize:
            for col in self.data.columns:
                if metric in col:
                    available_metrics.append(col)
        return available_metrics

    def _filter_by_configs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function filters out all those configurations that are not
        present in the configs_to_visualize list.

        Args:
            df: DataFrame to filter.

        Returns:
            DataFrame: Filtered DataFrame containing only the specified configurations.        
        """
        if not self.settings.configs_to_visualize:
            return df
        return df[df["config_hash"].isin(self.settings.configs_to_visualize)]
