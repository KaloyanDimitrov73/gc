from typing import List, Union, Optional, ClassVar
from typing_extensions import override
import weave
from weave.flow.scorer import auto_summarize
from pydantic import model_validator

from core import ProgressHandler
from core import AdditionalConfigParameter, RestrictionType
from experimentation.evaluation.base.evaluator import Evaluator
from core.logging.logging import get_logger

logger = get_logger(__name__)


class BasicScoreEvaluator(Evaluator):
    """
    Evaluator that computes precision, recall and F1 score.

    The implementation is based on the formula: 
    Precision = \frac{TP}{TP + FP}
    Recall = \frac{TP}{TP + FN}
    provided by https://link.springer.com/article/10.1007/s44163-024-00175-8
    F1 = 2 * \frac{Precision * Recall}{Precision + Recall}
    F2 = 5 * \frac{Precision * Recall}{4 * Precision + Recall}
    provided by https://en.wikipedia.org/wiki/F-score
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        AdditionalConfigParameter(
            name="k",
            description="The number of retrieved contexts to consider for precision calculation.",
            param_type=int,
            param_restriction=RestrictionType.GREQ_THAN_MINUS_1,
            default_value=-1
        ),
        AdditionalConfigParameter(
            name="context_type",
            description="The type of context to consider for the evaluation.",
            param_type=str,
            available_values=["triple", "entity", "document"],
            default_value="triple"
        )
    ]

    _settings: dict = {}
    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_late(self) -> "BasicScoreEvaluator":
        """Initialize after model validation."""
        if not self._initialized:
            self._settings = AdditionalConfigParameter.validate_dict(
                self.ADDITIONAL_CONFIG_PARAMS, self.config.additional_params
            )
            if self._settings["k"] > 0:
                self.name = "BasicScoreAtKEvaluator_" + \
                    self._settings["context_type"]
            else:
                self.name = "BasicScoreEvaluator_" + \
                    self._settings["context_type"]
            self._initialized = True
        return self

    @override
    def get_metric_names(self) -> List[str]:
        if self._settings.get("context_type") == "document":
            if self._settings["k"] > 0:
                return [
                    f"precision@{self._settings["k"]}_documents",
                    f"recall@{self._settings["k"]}_documents",
                    f"f1@{self._settings["k"]}_documents",
                    f"f2@{self._settings["k"]}_documents"
                ]
            return [
                "precision_documents",
                "recall_documents",
                "f1_documents",
                "f2_documents"
            ]
        if self._settings["context_type"] == "entity":
            if self._settings["k"] > 0:
                return [
                    f"precision@{self._settings["k"]}_entities",
                    f"recall@{self._settings["k"]}_entities",
                    f"f1@{self._settings["k"]}_entities",
                    f"f2@{self._settings["k"]}_entities"
                ]
            return [
                "precision_entities",
                "recall_entities",
                "f1_entities",
                "f2_entities"
            ]
        if self._settings["k"] > 0:
            return [
                f"precision@{self._settings["k"]}_triples",
                f"recall@{self._settings["k"]}_triples",
                f"f1@{self._settings["k"]}_triples",
                f"f2@{self._settings["k"]}_triples"
            ]
        return [
            "precision_triples",
            "recall_triples",
            "f1_triples",
            "f2_triples"
        ]

    @weave.op()
    @override
    def score(
        self,
        output: Optional[dict],
        golden_answer: Optional[str] = None,
        golden_triples: Optional[List[str]] = None,
        golden_doc_chunks: Optional[List[str]] = None
    ) -> dict:
        """
        Computes accuracy by checking if the top context matches any golden triple/entity.
        """
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting BASIC score evaluation.")
        contexts = self._get_contexts(output)
        context_texts = self._get_context_texts(contexts)

        if self._settings["k"] > 0:
            context_filtered = context_texts[:self._settings["k"]] if context_texts else [
            ]
        else:
            context_filtered = context_texts

        # If the context type is document based
        if self._settings.get("context_type") == "document":
            results = self._calculate_for_documents(
                context_filtered, golden_doc_chunks)

            if self._settings["k"] > 0:
                return {
                    f"precision@{self._settings['k']}_documents": results["precision"],
                    f"recall@{self._settings['k']}_documents": results["recall"],
                    f"f1@{self._settings['k']}_documents": results["f1"],
                    f"f2@{self._settings['k']}_documents": results["f2"]
                }
            return {
                "precision_documents": results["precision"],
                "recall_documents": results["recall"],
                "f1_documents": results["f1"],
                "f2_documents": results["f2"]
            }

        # Else if the context type is entity or triple based
        results = self._calculate(context_filtered, golden_triples)

        if self._settings.get("context_type") == "entity":
            if self._settings["k"] > 0:
                return {
                    f"precision@{self._settings['k']}_entities": results["precision"],
                    f"recall@{self._settings['k']}_entities": results["recall"],
                    f"f1@{self._settings['k']}_entities": results["f1"],
                    f"f2@{self._settings['k']}_entities": results["f2"]
                }
            return {
                "precision_entities": results["precision"],
                "recall_entities": results["recall"],
                "f1_entities": results["f1"],
                "f2_entities": results["f2"]
            }

        if self._settings["k"] > 0:
            return {
                f"precision@{self._settings['k']}_triples": results["precision"],
                f"recall@{self._settings['k']}_triples": results["recall"],
                f"f1@{self._settings['k']}_triples": results["f1"],
                f"f2@{self._settings['k']}_triples": results["f2"]
            }
        return {
            "precision_triples": results["precision"],
            "recall_triples": results["recall"],
            "f1_triples": results["f1"],
            "f2_triples": results["f2"]
        }

    @weave.op()
    @override
    def summarize(self, score_rows) -> dict:
        summarization = auto_summarize(score_rows)
        if not summarization:
            return {}

        precision_keys = [key for key in summarization.keys()
                          if "precision" in key]
        recall_keys = [key for key in summarization.keys() if "recall" in key]

        # Simply averaging the F1 and F2 scores does not provide the expected results
        # as it underestimates the scores. Therefore we need to calculate the F1 and F2 scores
        # based on the precision and recall scores.
        for i, f1_key in enumerate([key for key in summarization.keys() if "f1" in key]):
            if i < len(precision_keys) and i < len(recall_keys):
                precision = summarization[precision_keys[i]]["mean"]
                recall = summarization[recall_keys[i]]["mean"]
                if precision + recall > 0:
                    summarization[f1_key] = {"mean": 2 *
                                             (precision * recall) / (precision + recall)}
                else:
                    summarization[f1_key] = {"mean": 0.0}

        for i, f2_key in enumerate([key for key in summarization.keys() if "f2" in key]):
            if i < len(precision_keys) and i < len(recall_keys):
                precision = summarization[precision_keys[i]]["mean"]
                recall = summarization[recall_keys[i]]["mean"]
                if precision + recall > 0:
                    summarization[f2_key] = {"mean": 5 *
                                             (precision * recall) / ((4 * precision) + recall)}
                else:
                    summarization[f2_key] = {"mean": 0.0}

        row_scores = []
        for row in score_rows:
            row_dict = {}
            for key, value in row.items():
                row_dict[key] = value
            row_scores.append(row_dict)
        summarization["row_scores"] = row_scores

        return summarization

    def _calculate_for_documents(self, contexts: List[str], golden: Union[str, List[str]]) -> dict:
        """
        When scoring the document contexts, we need to consider, that the golden texts are sentences.
        The retrieved context however, can be multiple sentences. Therefore we check on at 
        sentence level, whether the golden sentence is in the retrieved context.
        """
        if golden is None or golden == "":
            if not contexts:
                return {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "f2": 1.0
                }
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "f2": 0.0
            }

        if not isinstance(golden, list):
            golden = [golden]

        context_set = set(contexts)
        golden_set = set(golden)

        tp = 0
        for context in context_set:
            for golden_chunk in golden_set:
                if golden_chunk in context:
                    tp += 1
                    break

        fp = len(context_set) - tp
        fn = len(golden_set) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if precision + recall > 0 else 0.0
        f2 = 5 * (precision * recall) / ((4 * precision) +
                                         recall) if precision + recall > 0 else 0.0

        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "f2": round(f2, 2)
        }

    def _calculate(self, contexts: List[str], golden: Union[str, List[str]]) -> dict:
        if golden is None or golden == "":
            if not contexts:
                return {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "f2": 1.0
                }
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "f2": 0.0
            }

        if not isinstance(golden, list):
            golden = [golden]

        if self._settings["context_type"] == "entity":
            golden = self._get_entities(golden)
            contexts = self._get_entities(contexts)

        context_set = set(contexts)
        golden_set = set(golden)

        # The intersection of the two sets
        tp = len(context_set & golden_set)
        # The difference between the two sets
        fp = len(context_set - golden_set)
        # The difference between the two sets
        fn = len(golden_set - context_set)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if precision + recall > 0 else 0.0
        f2 = 5 * (precision * recall) / ((4 * precision) +
                                         recall) if precision + recall > 0 else 0.0

        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "f2": round(f2, 2)
        }
