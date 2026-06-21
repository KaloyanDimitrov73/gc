from typing import Optional, List
from typing_extensions import override
import weave
from weave.flow.scorer import auto_summarize
from rouge_score import rouge_scorer

from core import ProgressHandler
from experimentation.evaluation.base.evaluator import Evaluator
from core.logging.logging import get_logger

logger = get_logger(__name__)


class RougeScoreEvaluator(Evaluator):
    """
    Implementation of the Rouge score using the Rouge-Score library on PyPi.
    https://pypi.org/project/rouge-score/
    """

    @override
    def get_metric_names(self) -> List[str]:
        return [
            "rouge_1_f1", "rouge_1_precision", "rouge_1_recall",
            "rouge_2_f1", "rouge_2_precision", "rouge_2_recall",
            "rouge_L_f1", "rouge_L_precision", "rouge_L_recall"
        ]

    @weave.op()
    @override
    def score(self,
              output: Optional[dict],
              golden_answer: Optional[str] = None,
              golden_triples: Optional[list[str]] = None,
              golden_doc_chunks: Optional[list[str]] = None) -> dict:
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting Rouge score evaluation.")
        if "generated_answer" not in output:
            logger.debug("No generated answer found in output.")
            return {
                "rouge_1_f1": 0.0,
                "rouge_1_precision": 0.0,
                "rouge_1_recall": 0.0,
                "rouge_2_f1": 0.0,
                "rouge_2_precision": 0.0,
                "rouge_2_recall": 0.0,
                "rouge_L_f1": 0.0,
                "rouge_L_precision": 0.0,
                "rouge_L_recall": 0.0
            }

        result = self._score_interal(output["generated_answer"], golden_answer)
        return result

    @weave.op()
    @override
    def summarize(self, score_rows) -> dict:
        summarization = auto_summarize(score_rows)
        if not summarization:
            return {}

        # Recalculate F1 scores from averaged precision and recall
        rouge_types = ["rouge_1", "rouge_2", "rouge_L"]

        for rouge_type in rouge_types:
            precision_key = f"{rouge_type}_precision"
            recall_key = f"{rouge_type}_recall"
            f1_key = f"{rouge_type}_f1"

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

    def _score_interal(self, generated_answer: str, golden_answer: str) -> dict:
        cleaned_answer = self._clean_answer(generated_answer)
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(golden_answer, cleaned_answer)

        return {
            "rouge_1_f1": round(scores["rouge1"].fmeasure, 2),
            "rouge_1_precision": round(scores["rouge1"].precision, 2),
            "rouge_1_recall": round(scores["rouge1"].recall, 2),
            "rouge_2_f1": round(scores["rouge2"].fmeasure, 2),
            "rouge_2_precision": round(scores["rouge2"].precision, 2),
            "rouge_2_recall": round(scores["rouge2"].recall, 2),
            "rouge_L_f1": round(scores["rougeL"].fmeasure, 2),
            "rouge_L_precision": round(scores["rougeL"].precision, 2),
            "rouge_L_recall": round(scores["rougeL"].recall, 2)
        }
