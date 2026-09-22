from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import os
import pandas as pd
import seaborn as sns
from sqa_system.core.logging.logging import get_logger

from .plotter.base.base_plotter import PlotterSetings
from .plotter.average_metrics_per_config_plotter import AverageMetricsPerConfigPlotter
from .plotter.metric_boxplots_plotter import MetricBoxplotsPlotter
from .plotter.metric_correlation_plotter import MetricCorrelationPlotter
from .plotter.metric_distributions_plotter import MetricDistributionsPlotter
from .plotter.metrics_table_plotter import MetricsTablePlotter
from .plotter.average_metrics_by_column_plotter import AverageMetricsByColumnPlotter
from .plotter.metric_by_column_per_config_plotter import MetricByColumnPerConfigPlotter
from .plotter.metrics_by_column_table_plotter import MetricsByColumnTablePlotter
from .plotter.sustainability_table_plotter import SustainabilityTablePlotter

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


class PlotType(Enum):
    """
    Enum class for the plots that are supported.
    """
    METRIC_DISTRIBUTIONS = "metric_distributions"
    METRIC_BOXPLOTS = "metric_boxplots"
    AVERAGE_METRICS_PER_CONFIG = "average_metrics_per_config"
    AVERAGE_METRICS_BY_COLUMN = "average_metrics_by_column"
    METRIC_BY_COLUMN_PER_CONFIG = "metric_by_column_per_config"
    METRIC_CORRELATION = "metric_correlation"
    TABLE = "table"
    METRIC_BY_COLUMN_TABLE = "metric_by_column_table"
    SUSTAINABILITY_TABLE = "sustainability_table"


@dataclass
class ExperimentVisualizerSettings:
    """
    Used to store the settings for the ExperimentVisualizer.

    Args:
        data_folder_path (str): The path to the folder containing the experiment data files.
            If provided, all data files in the folder and its subfolders will be used for 
            visualization.
        save_folder_path (str): The path to the folder where the visualizations will be saved.
        data_file_path (str): If provided, a single data file will be used for visualization,
            which is prioritized over the data_folder_path.
        qa_file_path (str): The path to the original QA file that was used to run the experiment.
            Not the prediction file! This is used to match the ids of the questions to the 
            prediction files to add additional columns.
        metrics (Dict[str, List[str]]): The list of the names of the metrics in the dataset to visualize.
            Should be a dict of lists, where each sublist contains the names of the metrics
            that should be visualized together and the key is the name of the group.
        configs_to_visualize (List[str]): The list of config hashes in the provided data
            that should be used for the visualization. If not provided all configs in the data
            are displayed in the created plots.
        should_save_to_file (bool): If True, the visualizations will be saved to the specified 
            folder.
        file_type (str): The file type to save the visualizations as (e.g., 'pdf', 'png').
        should_print (bool): If True, the visualizations will be printed to the console.
        config_to_name_mapping (Dict[str, str]): A mapping of config hashes to names that will be
            used to replace the hashes in the visualizations.
        new_metric_names_mapping (Dict[str, str]): A mapping of metric names to new names that will
            be used to replace the metric names in the visualizations.
        baseline_config (str): The config hash of the baseline config that will be used to
            highlight the baseline in the visualizations.
        round_decimals (int): If the plotter supports it, the metrics will be rounded to this number
            of decimals.
    """
    data_folder_path: Optional[str] = None
    save_folder_path: Optional[str] = None
    data_file_path: Optional[str] = None
    qa_file_path: Optional[str] = None
    metrics: Optional[Dict[str, List[str]]] = None
    configs_to_visualize: Optional[List[str]] = None
    should_save_to_file: bool = True
    file_type: Optional[str] = "pdf"
    should_print: bool = False
    config_to_name_mapping: Optional[Dict[str, str]] = None
    new_metric_names_mapping: Optional[Dict[str, str]] = None
    baseline_config: Optional[str] = None
    round_decimals: int = 3


class ExperimentVisualizer:
    """
    The ExperimentVisualizer class is used to visualize the results of experiments.

    Args:
        experiment_settings (ExperimentVisualizerSettings): The settings for the ExperimentVisualizer.
    """

    def __init__(
        self,
        experiment_settings: ExperimentVisualizerSettings
    ):
        self.file_type = experiment_settings.file_type
        self.should_save_to_file = experiment_settings.should_save_to_file
        self.should_print = experiment_settings.should_print
        self.baseline_config = experiment_settings.baseline_config
        self.data: Dict[str, pd.DataFrame] = {}
        self.save_folder_path = None
        self.round_decimals = experiment_settings.round_decimals
        self.configs_to_visualize = experiment_settings.configs_to_visualize or None
        self._load_data(experiment_settings)
        self._replace_hashes_with_names(experiment_settings)
        self._prepare_base_metrics(experiment_settings)
        self._replace_metric_names(experiment_settings)

    def _load_data(self, experiment_settings: ExperimentVisualizerSettings):
        """
        Loads the data from the provided file or folder path.
        If a file path is provided, it will be used to load the data.
        If a folder path is provided, all CSV files in the folder and its subfolders
        will be loaded into a dictionary.

        Args:
            experiment_settings (ExperimentVisualizerSettings): The settings for the ExperimentVisualizer.
        """
        potential_save_path = ""
        # Load data from file or folder
        if experiment_settings.data_file_path:
            potential_save_path = os.path.dirname(
                experiment_settings.data_file_path)
            self.data = self._prepare_data(
                df=pd.read_csv(experiment_settings.data_file_path),
                qa_file_path=experiment_settings.qa_file_path)
        elif experiment_settings.data_folder_path:
            potential_save_path = experiment_settings.data_folder_path
            self.data = self._load_data_from_folder(
                folder_path=experiment_settings.data_folder_path,
                qa_file_path=experiment_settings.qa_file_path)
        else:
            raise ValueError(
                "Either data_folder_path or data_file_path must be provided.")

        if experiment_settings.save_folder_path:
            self.save_folder_path = experiment_settings.save_folder_path
        else:
            self.save_folder_path = potential_save_path

    def _replace_hashes_with_names(self, experiment_settings: ExperimentVisualizerSettings):
        """
        If the user provides a mapping of config hashes to names, use it to replace the hashes

        Args:
            experiment_settings (ExperimentVisualizerSettings): The settings for the ExperimentVisualizer.
        """
        if experiment_settings.config_to_name_mapping and self.data:
            current_keys = list(self.data.keys())
            for current_hash in current_keys:
                if current_hash in experiment_settings.config_to_name_mapping:
                    new_name = experiment_settings.config_to_name_mapping[current_hash]
                    df = self.data.pop(current_hash)
                    df["config_hash"] = new_name
                    self.data[new_name] = df
                else:
                    logger.warning(
                        "Config hash %s not found in mapping.", current_hash)

    def _replace_metric_names(self, experiment_settings: ExperimentVisualizerSettings):
        """
        If the user provides a mapping of metric names to new names, use it to replace the metric names.

        Args:
            experiment_settings (ExperimentVisualizerSettings): The settings for the ExperimentVisualizer.
        """
        if not experiment_settings.new_metric_names_mapping:
            return
        
        if experiment_settings.new_metric_names_mapping and self.data:
            for df in self.data.values():
                for old_name, new_name in experiment_settings.new_metric_names_mapping.items():
                    if old_name in df.columns:
                        df.rename(columns={old_name: new_name}, inplace=True)
                    else:
                        logger.warning(
                            "Metric name %s not found in DataFrame.", old_name)
        
        for category, metrics_list in self.base_metrics.items():
            updated_metrics = []
            for metric in metrics_list:
                if metric in experiment_settings.new_metric_names_mapping:
                    updated_metrics.append(experiment_settings.new_metric_names_mapping[metric])
                else:
                    updated_metrics.append(metric)
            self.base_metrics[category] = updated_metrics

    def _prepare_base_metrics(self, experiment_settings: ExperimentVisualizerSettings):
        """
        Prepares those metrics for which to generate the plots.
        If no metrics are provided, the default metrics are used.

        Args:
            experiment_settings (ExperimentVisualizerSettings): The settings for the ExperimentVisualizer.
        """
        if not experiment_settings.metrics:
            self.base_metrics = {
                "Retrieval": [
                    "hit@10_triples",
                    "map@10_triples",
                    "mrr@10_triples",
                    "precision@10_triples",
                    "recall@10_triples",
                    "f1@10_triples",
                    "f2@10_triples",
                    "precision_triples",
                    "recall_triples",
                    "f1_triples",
                    "f2_triples",
                    "exact_match@10_triples",
                    "exact_match_triples",
                    "context_entity_recall",
                    "context_recall",
                    "llm_context_precision_with_reference",
                    "llm_context_precision_without_reference"
                ],
                "Generation": [
                    "answer_relevancy",
                    "bleu",
                    "rouge",
                    "non_llm_string_similarity",
                    "factual_correctness",
                    "faithfulness",
                    "bertscore",
                    "semantic_similarity",
                    "instruction_following"
                ],
                "Runtime": [
                    "runtime",
                ],
                "LLM Cost": [
                    "llm_cost",
                ],
                "LLM Tokens": [
                    "llm_tokens",
                ],
                "Energy Consumption": [
                    "energy_consumption",
                ],
                "Emissions": [
                    "emissions"
                ]
            }
        else:
            self.base_metrics = experiment_settings.metrics

    def run(self,
            plots_to_generate: List[PlotType] = None,
            column_to_group_by: Optional[str] = None):
        """
        Runs the visualization process by generating plots
        and saving them to the specified folder.

        Args:
            plots_to_generate (List[PlotType]): The list of plots to generate.
            column_to_group_by (str): This is an optional parameter that is only
                used for the AverageMetricsByColumnPlotter and the
                MetricByColumnPerConfigPlotter. It specifies the column
                by which to group the data for visualization.
        """
        if not self.data:
            logger.error("No valid data found. Aborting visualization.")
            return

        # Combine data for easier plotting with external classes
        combined_data = self._get_combined_data()
        if combined_data is None or combined_data.empty:
            logger.warning("No combined data to visualize.")
            return

        # Determine which plots to generate
        if not plots_to_generate or len(plots_to_generate) == 0:
            plots_to_generate = [
                PlotType.METRIC_DISTRIBUTIONS,
                PlotType.AVERAGE_METRICS_PER_CONFIG,
                PlotType.TABLE,
            ]

        # Map each PlotType to the corresponding plotter class
        plotter_classes = {
            PlotType.METRIC_DISTRIBUTIONS: MetricDistributionsPlotter,
            PlotType.METRIC_BOXPLOTS: MetricBoxplotsPlotter,
            PlotType.AVERAGE_METRICS_PER_CONFIG: AverageMetricsPerConfigPlotter,
            PlotType.METRIC_CORRELATION: MetricCorrelationPlotter,
            PlotType.TABLE: MetricsTablePlotter,
            PlotType.AVERAGE_METRICS_BY_COLUMN: AverageMetricsByColumnPlotter,
            PlotType.METRIC_BY_COLUMN_PER_CONFIG: MetricByColumnPerConfigPlotter,
            PlotType.METRIC_BY_COLUMN_TABLE: MetricsByColumnTablePlotter,
            PlotType.SUSTAINABILITY_TABLE: SustainabilityTablePlotter,
        }

        self._generate_plots(
            plots_to_generate=plots_to_generate,
            plotter_classes=plotter_classes,
            combined_data=combined_data,
            column_to_group_by=column_to_group_by
        )

    def _generate_plots(self,
                        plots_to_generate: List[PlotType],
                        plotter_classes: Dict[PlotType, type],
                        combined_data: pd.DataFrame,
                        column_to_group_by: Optional[str] = None):
        """
        Internal function to loop over the provided plots and run the generations.
        """
        if self.save_folder_path:
            vis_folder = self.save_folder_path
            os.makedirs(vis_folder, exist_ok=True)
        else:
            vis_folder = None

        for plot_type in plots_to_generate:
            plotter_class = plotter_classes.get(plot_type)
            if not plotter_class:
                logger.warning("Invalid plot type: %s", plot_type)
                continue

            if plot_type in {PlotType.TABLE, PlotType.METRIC_DISTRIBUTIONS}:
                all_metrics = []
                for _, metric_group in self.base_metrics.items():
                    all_metrics.extend(metric_group)
                plotter = plotter_class(
                    data=combined_data,
                    base_metrics=all_metrics,
                    settings=PlotterSetings(
                        should_save_to_file=self.should_save_to_file,
                        should_print=self.should_print,
                        file_type=self.file_type,
                        configs_to_visualize=self.configs_to_visualize,
                        baseline_config=self.baseline_config,
                        round_decimals=self.round_decimals
                    )
                )
                self._run_plotter(plotter, vis_folder, plot_type)
                continue

            if plot_type in {PlotType.METRIC_BY_COLUMN_TABLE, PlotType.SUSTAINABILITY_TABLE}:
                all_metrics = []
                for _, metric_group in self.base_metrics.items():
                    all_metrics.extend(metric_group)

                plotter = plotter_class(
                    data=combined_data,
                    base_metrics=all_metrics,
                    settings=PlotterSetings(
                        should_save_to_file=self.should_save_to_file,
                        should_print=self.should_print,
                        file_type=self.file_type,
                        configs_to_visualize=self.configs_to_visualize,
                        baseline_config=self.baseline_config,
                        round_decimals=self.round_decimals
                    ),
                    column_name=column_to_group_by
                )
                self._run_plotter(plotter, vis_folder, plot_type)
                continue

            for metric_group_key, metric_group in self.base_metrics.items():
                if plot_type in {PlotType.AVERAGE_METRICS_BY_COLUMN,
                                 PlotType.METRIC_BY_COLUMN_PER_CONFIG}:
                    if column_to_group_by is None:
                        logger.error(
                            "Column to group by must be provided for %s plot.", plot_type.name)
                        continue
                    plotter = plotter_class(
                        data=combined_data,
                        base_metrics=metric_group,
                        settings=PlotterSetings(
                            should_save_to_file=self.should_save_to_file,
                            should_print=self.should_print,
                            file_type=self.file_type,
                            configs_to_visualize=self.configs_to_visualize,
                            baseline_config=self.baseline_config,
                            round_decimals=self.round_decimals
                        ),
                        group_name=metric_group_key,
                        column_name=column_to_group_by
                    )
                    self._run_plotter(plotter, vis_folder, plot_type)
                    continue

                plotter = plotter_class(
                    data=combined_data,
                    base_metrics=metric_group,
                    settings=PlotterSetings(
                        should_save_to_file=self.should_save_to_file,
                        should_print=self.should_print,
                        file_type=self.file_type,
                        configs_to_visualize=self.configs_to_visualize,
                        baseline_config=self.baseline_config,
                        round_decimals=self.round_decimals
                    ),
                    group_name=metric_group_key
                )
                self._run_plotter(plotter, vis_folder, plot_type)
        logger.info("Visualizations saved in %s", vis_folder)

    def _run_plotter(self, plotter, vis_folder: Optional[str], plot_type: PlotType):
        try:
            plotter.plot(vis_folder)
            logger.debug(
                "Successfully generated %s visualization.", plot_type.name
            )
        except Exception as e:
            logger.error(
                "Error generating %s visualization: %s", plot_type.name, str(
                    e)
            )

    def _load_data_from_folder(self,
                               folder_path: str,
                               qa_file_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load CSV files from the given folder and its subfolders into a dictionary.
        Key = config_hash, Value = DataFrame

        Args:
            folder_path (str): The path to the folder containing the CSV files.
            qa_file_path (str): The path to the original QA file that was used to run the experiment.
                Not the prediction file! This is used to match the ids of the questions to the
                prediction files to add additional columns.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where the keys are the unique config_hash values
                and the values are the corresponding DataFrames.
        """
        data_dict = {}
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        if df.empty:
                            logger.warning("Empty CSV file: %s", file_path)
                            continue
                        prepared_data = self._prepare_data(df)
                        if prepared_data:
                            data_dict = self._merge_data(
                                data_dict, prepared_data)
                        else:
                            logger.warning(
                                "Invalid data structure in %s", file_path)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                        logger.error(
                            "Error parsing CSV file %s: %s", file_path, str(e))
                    except IOError as e:
                        logger.error("Error reading file %s: %s",
                                     file_path, str(e))

        if not data_dict:
            logger.warning("No valid data files found.")
        else:
            logger.info("Successfully loaded %d valid CSV files.",
                        len(data_dict))

        # If the QA dataset file path is provided, merge the dataframes by question id
        if qa_file_path:
            data_dict = self._merge_by_question_id(
                qa_file_path, data_dict)
        return data_dict

    def _merge_data(self,
                    data_dict: Dict[str, pd.DataFrame],
                    prepared_data: Dict[str, pd.DataFrame]):
        # Merging dictionaries
        for k, v in prepared_data.items():
            if k in data_dict:
                data_dict[k] = pd.concat(
                    [data_dict[k], v], ignore_index=True
                )
            else:
                data_dict[k] = v

        return data_dict

    def _prepare_data(self,
                      df: pd.DataFrame,
                      qa_file_path: Optional[str] = None) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Prepare the data for visualization by splitting the DataFrame into separate
        DataFrames for each unique config_hash value.

        Args:
            df (pd.DataFrame): The DataFrame to prepare.
            qa_file_path (str): The path to the original QA file that was used to run the experiment.
                Not the prediction file! This is used to match the ids of the questions to the
                prediction files to add additional columns.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where the keys are the unique config_hash values
                and the values are the corresponding DataFrames.
        """
        if not self._validate_data(df):
            return None

        result = {}
        # Group the DataFrame by config_hash and split into separate DataFrames
        for config_hash, group_df in df.groupby("config_hash"):
            result[config_hash] = group_df.reset_index(drop=True)

        # If the QA dataset file path is provided, merge the dataframes by question id
        if qa_file_path:
            result = self._merge_by_question_id(
                qa_file_path, result)

        return result

    def _merge_by_question_id(self,
                              qa_dataset_file_path: str,
                              datas: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Loads the qa dataset from the path and merges it with the dataframes
        by the ids of the question. For this to work, the predicition dataframes
        must have a 'uid' key and the qa dataset path must have it as well.

        Args:
            qa_dataset_file_path (str): The path to the QA dataset file.
            datas (Dict[str, pd.DataFrame]): The dictionary of DataFrames to merge.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where the keys are the unique config_hash values
                and the values are the corresponding DataFrames with additional columns from the QA dataset.
        """
        if not os.path.exists(qa_dataset_file_path):
            logger.error(
                "Could not find the QA datset file path: %s", qa_dataset_file_path)
            return datas

        qa_df = pd.read_csv(qa_dataset_file_path)

        if "uid" not in qa_df.columns:
            logger.error("The QA dataset does not contain a 'uid' column.")
            return datas

        qa_cols_set = set(qa_df.columns)
        qa_cols_set.remove("uid")
        processed_datas = {}
        for config_hash, df in datas.items():
            if "uid" not in df.columns:
                logger.warning(
                    "The prediction dataset for hash '%s' does not contain a 'uid' column. Skipping merge for this dataset.", config_hash)
                processed_datas[config_hash] = df
                continue

            df_cols_set = set(df.columns)
            df_cols_set.remove("uid")
            overlap_cols = list(df_cols_set.intersection(qa_cols_set))
            df_temp = df.drop(columns=overlap_cols, errors="ignore")

            try:
                merged_df = pd.merge(df_temp, qa_df, on="uid", how="left")
                processed_datas[config_hash] = merged_df
            except Exception as e:
                logger.error(
                    "Error merging dataframe for hash '%s': %s", config_hash, e)
                processed_datas[config_hash] = df

        return processed_datas

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Check if the data is valid for the experiment visualizer."""
        return "config_hash" in data.columns

    def _get_combined_data(self) -> Optional[pd.DataFrame]:
        """Combine all data into a single DataFrame for plotting."""
        if not self.data:
            return None

        combined_data = pd.concat(self.data.values(), ignore_index=True)
        if combined_data.empty:
            logger.warning("Filtered data is empty after combining.")
            return None
        return combined_data
