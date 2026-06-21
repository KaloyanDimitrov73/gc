import os
from typing import List, Optional, ClassVar
import random
import time
import tempfile
import filelock
from typing_extensions import override
import weave
from weave.flow.scorer import auto_summarize
from pydantic import model_validator
from evaluate import load

from core import ProgressHandler
from core import AdditionalConfigParameter
from experimentation.evaluation.base.evaluator import Evaluator
from core.logging.logging import get_logger

logger = get_logger(__name__)


class BertScoreEvaluator(Evaluator):
    """
    Evaluator that computes the BertScore metrics on the generated answer 
    and the golden answer.
    This implementation uses the huggingface evaluation:
    https://huggingface.co/spaces/evaluate-metric/bertscore
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        AdditionalConfigParameter(
            name="model_type",
            description="BERT model type for BERTScore evaluation.",
            param_type=str,
            default_value="distilbert-base-uncased"
        )
    ]

    _settings: dict = {}
    _initialized: bool = False
    _scorer = None

    @model_validator(mode='after')
    def initialize_late(self) -> 'BertScoreEvaluator':
        """Initialize after model validation"""
        if not self._initialized:
            self._settings = AdditionalConfigParameter.validate_dict(
                self.ADDITIONAL_CONFIG_PARAMS, self.config.additional_params)
        return self

    @override
    def get_metric_names(self) -> List[str]:
        return [
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1"
        ]

    @weave.op()
    @override
    def summarize(self, score_rows) -> dict:
        summarization = auto_summarize(score_rows)
        if not summarization:
            return {}

        # Recalculate F1 score from averaged precision and recall
        precision_key = "bertscore_precision"
        recall_key = "bertscore_recall"
        f1_key = "bertscore_f1"

        if precision_key in summarization and recall_key in summarization:
            precision = summarization[precision_key]["mean"]
            recall = summarization[recall_key]["mean"]

            if precision + recall > 0:
                summarization[f1_key] = {"mean": 2 * \
                    (precision * recall) / (precision + recall)}
            else:
                summarization[f1_key] = {"mean": 0.0}

        # Add individual row scores
        row_scores = []
        for row in score_rows:
            row_dict = {}
            for key, value in row.items():
                row_dict[key] = value
            row_scores.append(row_dict)
        summarization["row_scores"] = row_scores

        return summarization

    @weave.op()
    @override
    def score(self,
              output: Optional[dict],
              golden_answer: Optional[str] = None,
              golden_triples: Optional[list[str]] = None,
              golden_doc_chunks: Optional[list[str]] = None) -> dict:
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting BERT score evaluation.")
        if golden_answer is None or golden_answer == "":
            return {
                "bertscore_precision": 1.0,
                "bertscore_recall": 1.0,
                "bertscore_f1": 1.0
            }
        if "generated_answer" not in output:
            logger.debug("No generated answer found in output.")
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0
            }

        return self._score_interal(output["generated_answer"], golden_answer)

    def _score_interal(self, generated_answer: str, golden_answer: str) -> dict:

        # Because when using the BERT scorer it temporarily creates a file
        # that needs to be accessed for the scorer to work. This can't be done
        # when running in parallel so we use this to ensure that the file is only
        # accessed by one process at a time.
        # Code based on: https://stackoverflow.com/questions/489861/locking-a-file-in-python
        lock_file = os.path.join(tempfile.gettempdir(), "bertscore_lock.lock")
        max_retries = 10
        retry_count = 0
        
        cleaned_answer = self._clean_answer(generated_answer)

        while retry_count < max_retries:
            try:
                with filelock.FileLock(lock_file, timeout=120):
                    # Load the scorer inside the locked section
                    scorer = load("bertscore")
                    results = scorer.compute(
                        predictions=[cleaned_answer],
                        references=[golden_answer],
                        model_type=self._settings["model_type"],
                        device="cpu"
                    )

                    return {
                        "bertscore_precision": results["precision"][0],
                        "bertscore_recall": results["recall"][0],
                        "bertscore_f1": results["f1"][0]
                    }
            except filelock.Timeout:
                retry_count += 1
                logger.warning(
                    ("Encountered a timeout with Bert Score "
                     f"Evaluator (attempt {retry_count}/{max_retries})"))

                if retry_count >= max_retries:
                    logger.error(
                        "Bert Evaluator failed after multiple timeouts")
                    return {
                        "bertscore_precision": 0.0,
                        "bertscore_recall": 0.0,
                        "bertscore_f1": 0.0,
                    }
                backoff_time = 2 ** retry_count
                jitter = backoff_time * 0.3
                backoff_time += random.uniform(-jitter, jitter)
                backoff_time = min(backoff_time, 120)
                logger.info(f"Backing off for {backoff_time:.2f} seconds")
                time.sleep(backoff_time)
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }
