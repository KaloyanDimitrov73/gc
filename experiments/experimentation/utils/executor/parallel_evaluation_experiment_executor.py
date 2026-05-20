import asyncio
import concurrent.futures
from typing import List, Dict, Any
from typing_extensions import override

from sqa_system.core.config.models.pipeline_config import PipelineConfig
from sqa_system.core.data.pipeline_data_collector import PipelineData
from sqa_system.core.logging.logging import get_logger
from .base.base_experiment_executor import BaseExperimentExecutor
from ..experiment_evaluator import ExperimentEvaluator

logger = get_logger(__name__)


class ParallelEvaluationExperimentExecutor(BaseExperimentExecutor):
    """
    Executes pipelines sequentially, and allows evaluators to run concurrently
    but only one at a time.

    This allows pipelines to start even before the previous evaluation has finished
    while ensuring that only one evaluation is running at a time.
    """

    @override
    def execute_experiments(self,
                            results_folder_path: str,
                            configs: List[PipelineConfig],
                            visualize_results:bool) -> List[Dict[str, Any]]:
        all_results = []
        pending_evaluations = []

        # Executor with a single worker to ensure evaluations run one at a time
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for config in configs:
                # Run the pipeline and get its result
                pipeline_results = self._run_pipeline(config)
                self.progress_handler.update_task_by_string_id("configuration")

                # Submit the evaluation
                future = executor.submit(
                    self._evaluate_results,
                    pipeline_results,
                    config
                )
                pending_evaluations.append((future, config))

            # Process all evaluation futures
            for future, config in pending_evaluations:
                try:
                    eval_results = future.result()
                    
                    processed_results = self.result_processor.process_result_rows(
                        pipeline_results,
                        self.prepared_qa_dataset,
                        config,
                        eval_results
                    )
                    all_results.extend(processed_results)

                    result_df = self.result_processor.create_result_dataframe(
                        processed_results,
                        self.prepared_qa_dataset_df
                    )

                    self._save_results(
                        results_folder_path=results_folder_path,
                        config=config,
                        data_to_save=result_df
                    )
                except Exception as e:
                    logger.error(
                        f"Error in evaluation for config {config.config_hash}: {e}"
                    )
                    
        if visualize_results:
            self._visualize_results(
                results_folder_path=results_folder_path,
                baseline_config_hash=configs[0].config_hash if len(configs) > 1 else None
            )

        return all_results

    def _evaluate_results(self,
                          pipeline_results: List[PipelineData],
                          config: PipelineConfig):
        """Helper function to evaluate results asynchronously"""
        evaluator = ExperimentEvaluator()
        model_name = None
        for pipe in config.pipes:
            if "kg_retrieval" in pipe.type or "document_retrieval" in pipe.type:
                model_name = pipe.retriever_type
                break
        return asyncio.run(evaluator.evaluate_results_with_weave(
            qa_dataset_dict=self.prepared_qa_dataset,
            evaluators=self.evaluators,
            pipeline_results=pipeline_results,
            model_name=model_name
        ))
