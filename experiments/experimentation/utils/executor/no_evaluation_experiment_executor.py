from typing import List, Dict, Any
from typing_extensions import override

from sqa_system.core.config.models.pipeline_config import PipelineConfig
from sqa_system.core.logging.logging import get_logger
from .base.base_experiment_executor import BaseExperimentExecutor

logger = get_logger(__name__)


class NoEvaluationExperimentExecutor(BaseExperimentExecutor):
    """
    Executor that runs pipelines without evaluating them.
    """

    @override
    def execute_experiments(self,
                            results_folder_path: str,
                            configs: List[PipelineConfig],
                            visualize_results:bool) -> List[Dict[str, Any]]:
        experiment_results = []
        for config in configs:
            pipeline_results = self._run_pipeline(config)

            processed_results = self.result_processor.process_result_rows(
                pipeline_datas=pipeline_results,
                prepared_dataset=self.prepared_qa_dataset,
                config=config
            )
            
            result_df = self.result_processor.create_result_dataframe(
                row_results=processed_results,
                questions_df=self.prepared_qa_dataset_df
            )
            experiment_results.extend(processed_results)
            
            self._save_results(
                data_to_save=result_df,
                results_folder_path=results_folder_path,
                config=config
            )

            self.progress_handler.update_task_by_string_id("configuration")
            
        if visualize_results:
            self._visualize_results(
                results_folder_path=results_folder_path,
                baseline_config_hash=configs[0].config_hash if len(configs) > 1 else None
            )
        
        return experiment_results
