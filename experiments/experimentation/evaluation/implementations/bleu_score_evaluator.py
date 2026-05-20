from typing import Optional, List
from typing_extensions import override
import weave
from sacrebleu.metrics import BLEU

from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from experimentation.evaluation.base.evaluator import Evaluator
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


class BleuScoreEvaluator(Evaluator):
    """
    Implementation of the BLEU score using the Sacrebleu library on PyPi.
    https://pypi.org/project/sacrebleu/
    """

    @override
    def get_metric_names(self) -> List[str]:
        return [
            "bleu_score"
        ]

    @weave.op()
    @override
    def score(self,
              output: Optional[dict],
              golden_answer: Optional[str] = None,
              golden_triples: Optional[list[str]] = None,
              golden_doc_chunks: Optional[list[str]] = None) -> dict:
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting BLEU score evaluation.")
        if "generated_answer" not in output:
            logger.debug("No generated answer found in output.")
            return {
                "bleu_score": 0.0
            }

        return self._score_interal(output["generated_answer"], golden_answer)

    def _score_interal(self, generated_answer: str, golden_answer: str) -> dict:
        bleu_scorer = BLEU()
        cleaned_answer = self._clean_answer(generated_answer)
        score = bleu_scorer.corpus_score(
            [cleaned_answer], [[golden_answer]]).score / 100

        return {
            "bleu_score": round(score, 2)
        }
