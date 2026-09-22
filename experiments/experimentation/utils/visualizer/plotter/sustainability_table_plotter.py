from typing import Optional
import pandas as pd
import numpy as np

from sqa_system.core.logging.logging import get_logger

from .base.base_plotter import BasePlotter

logger = get_logger(__name__)


class SustainabilityTablePlotter(BasePlotter):
    """
    Generates a table with sustainability metrics according to Kaplan et al.
    https://publikationen.bibliothek.kit.edu/1000180194

    Args:
        column_name: The name of the metric to calculate the sustainability
            metrics by. For example F1.

    Note:
        This plotter requires both a 'emissions' and 'total_energy_consumption"
        column in the data. It then creates two different tables, one for the
        emissions and one for the total energy consumption.        
    """

    def __init__(self, column_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_name = column_name
        self.sustainability_columns = ["emissions", "total_energy_consumption"]

    def plot(self, vis_folder: Optional[str]):
        if self.data is None or self.data.empty:
            logger.warning("No data available to plot metrics table.")
            return

        required_base_cols = ["config_hash", self.column_name]
        required_sus_cols = self.sustainability_columns
        missing_cols = [col for col in required_base_cols +
                        required_sus_cols if col not in self.data.columns]
        if missing_cols:
            logger.warning(
                f"Missing required columns in data: {missing_cols}.")
            return

        available_metrics = self._get_available_metrics()
        if not available_metrics:
            logger.warning("No available metrics for plotting table.")
            return

        if self.column_name not in available_metrics:
            logger.warning(
                f"Column '{self.column_name}' not found in available metrics.")
            return

        # Calculate statistics for each metric per configuration
        for sus_metric_name in self.sustainability_columns:
            stats_list = self._calculate_statistics(sus_metric_name)
            if not stats_list:
                logger.warning(
                    f"Could not calculate basic statistics for {sus_metric_name}.")
                continue

            stats_df = pd.DataFrame(stats_list)
            if self.settings.should_save_to_file and vis_folder:
                table_dir = self.file_path_manager.combine_paths(
                    vis_folder,
                    "sustainability",
                    f"{sus_metric_name}_metrics_table.csv"
                )
                self.file_path_manager.ensure_dir_exists(table_dir)
                stats_df.to_csv(table_dir, index=False)

            if self.settings.should_print:
                print(stats_df)

    def _calculate_statistics(self, sustainability_metric: str) -> list[dict]:
        """
        The main function to calculate the sustainability metrics.
        It calculates the mean performance metric and the mean sustainability
        metric for each configuration. It also calculates the relative
        sustainability metric, the delta sustainability metric and the
        normalized sustainability metric.

        Args:
            sustainability_metric: The name of the sustainability metric to
                calculate. For example 'emissions' or 'total_energy_consumption'.

        Returns:
            A list of dictionaries containing the calculated metrics for each
            configuration.
        """

        required_cols = ["config_hash",
                         self.column_name, sustainability_metric]
        if not all(col in self.data.columns for col in required_cols):
            logger.warning(f"Missing required columns for {sustainability_metric} table. "
                           f"Need: {required_cols}. Found: {self.data.columns.tolist()}")
            return None

        # Filter data by configured configs
        filtered_data = self._filter_by_configs(self.data)

        stats_data = []
        for config, group in filtered_data.groupby("config_hash"):
            performance_values = group[self.column_name].astype(float)
            sustainability_values = group[sustainability_metric].astype(float)

            mean_performance_metric = performance_values.mean()
            mean_sustainability = sustainability_values.mean()

            stats_data.append({
                "Config": config,
                self.column_name: mean_performance_metric,
                sustainability_metric: mean_sustainability,
            })

        stats_data = self._calculate_relative_sustainability(
            stats_data, sustainability_metric)
        stats_data = self._calculate_delta_sustainability(
            stats_data, sustainability_metric)
        stats_data = self._add_normalized_metrics(
            stats_data, sustainability_metric)

        # Format numbers as strings
        for row in stats_data:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    row[k] = f"{v:.6f}"

        return stats_data

    def _calculate_relative_sustainability(self, stats_data: list[dict], sustainability_metric: str):
        """
        Calculates the relative sustainability metric for each configuration.

        Args:
            stats_data: A list of dictionaries containing the calculated sustainability
                and performance metrics for each configuration.
            sustainability_metric: The name of the sustainability metric to calculate.
                For example 'emissions' or 'total_energy_consumption'.
        """
        for row in stats_data:
            sustainability_value = row[sustainability_metric]
            performance_value = row[self.column_name]
            if performance_value == 0:
                row[f"{sustainability_metric}_rel"] = float("inf")
            else:
                row[f"{sustainability_metric}_rel"] = sustainability_value / \
                    performance_value
        return stats_data

    def _calculate_delta_sustainability(self, stats_data: list[dict], sustainability_metric: str):
        """
        Calculates the delta sustainability metric for each configuration.

        Args:
            stats_data: A list of dictionaries containing the calculated sustainability
                and performance metrics for each configuration.
            sustainability_metric: The name of the sustainability metric to calculate.
                For example 'emissions' or 'total_energy_consumption'.
        """
        baseline = min(stats_data, key=lambda r: r[self.column_name])
        baseline_perf = baseline[self.column_name]
        baseline_sus = baseline[sustainability_metric]

        for row in stats_data:
            if row["Config"] == baseline["Config"]:
                row[f"delta_{sustainability_metric}"] = 0.0
            else:
                cur_perf = row[self.column_name]
                cur_sus = row[sustainability_metric]
                if cur_sus == 0:
                    row[f"delta_{sustainability_metric}"] = float("inf")
                else:
                    row[f"delta_{sustainability_metric}"] = (
                        cur_perf - baseline_perf) * (baseline_sus / cur_sus)
        return stats_data

    def _add_normalized_metrics(self,
                                stats_data: list[dict],
                                sustainability_metric: str) -> list[dict]:
        """
        Normalizes the sustainability metrics for each configuration.

        Args:
            stats_data: A list of dictionaries containing the calculated sustainability
                and performance metrics for each configuration.
            sustainability_metric: The name of the sustainability metric to calculate.
                For example 'emissions' or 'total_energy_consumption'.
        """
        raw_key = sustainability_metric
        rel_key = f"{sustainability_metric}_rel"

        raw_vals = [row[raw_key] for row in stats_data]
        rel_vals = [row[rel_key]
                    for row in stats_data if np.isfinite(row[rel_key])]

        min_raw, max_raw = min(raw_vals), max(raw_vals)
        if rel_vals:
            min_rel, max_rel = min(rel_vals), max(rel_vals)
        else:
            min_rel = max_rel = None

        #   n(CE)      = 1 - (CE - CE_min) / (CE_max - CE_min)
        #   n(CE_rel)  = 1 - (CE_rel - CE_rel_min) / (CE_rel_max - CE_rel_min)
        for row in stats_data:
            ce = row[raw_key]
            # Here we normalize the raw sustainability metric
            if max_raw == min_raw:
                row[f"n({raw_key})"] = 1.0
            else:
                row[f"n({raw_key})"] = 1.0 - \
                    (ce - min_raw) / (max_raw - min_raw)

            # Here we normalize the relative sustainability metric
            cer = row.get(rel_key, np.nan)
            if min_rel is None or max_rel == min_rel or not np.isfinite(cer):
                row[f"n({rel_key})"] = 1.0 if cer == min_rel == max_rel else 0.0
            else:
                row[f"n({rel_key})"] = 1.0 - \
                    (cer - min_rel) / (max_rel - min_rel)

        return stats_data
