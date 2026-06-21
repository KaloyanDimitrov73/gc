from typing import List, Optional, ClassVar, Union
from typing_extensions import override
import weave
from pydantic import model_validator

from core import ProgressHandler
from core import AdditionalConfigParameter, RestrictionType
from experimentation.evaluation.base.evaluator import Evaluator
from core.logging.logging import get_logger

logger = get_logger(__name__)


class ExactMatchEvaluator(Evaluator):
    """
    Evaluator that computes Exact Match (EM) by checking the proportion of retrieved contexts
    that exactly match any of the expected golden triples/entities.

    The implementation is based on the formula: Exact Match = \frac{PEM}{N_{pred}}
    provided by https://link.springer.com/article/10.1007/s44163-024-00175-8
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
    def initialize_late(self) -> "ExactMatchEvaluator":
        """Initialize after model validation."""
        if not self._initialized:
            self._settings = AdditionalConfigParameter.validate_dict(
                self.ADDITIONAL_CONFIG_PARAMS, self.config.additional_params
            )
            if self._settings["k"] > 0:
                self.name = "ExactMatchAtKEvaluator_" + \
                    self._settings["context_type"]
            else:
                self.name = "ExactMatchEvaluator_" + \
                    self._settings["context_type"]
            self._initialized = True

        return self
    
    @override
    def get_metric_names(self) -> List[str]:
        if self._settings["context_type"] == "entity":
            return [
                f"exact_match@{self._settings['k']}_entities" if self._settings['k'] > 0 else "exact_match_entities"
            ]
        if self._settings["context_type"] == "document":
            return [
                f"exact_match@{self._settings['k']}_documents" if self._settings['k'] > 0 else "exact_match_documents"
            ]
        return [
            f"exact_match@{self._settings['k']}_triples" if self._settings['k'] > 0 else "exact_match_triples"
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
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting EXACT score evaluation.")
        contexts = self._get_contexts(output)
        context_texts = self._get_context_texts(contexts)

        if self._settings["k"] > 0:
            context_filtered = context_texts[:self._settings["k"]] if context_texts else [
            ]
        else:
            context_filtered = context_texts

        if self._settings.get("context_type") == "document":
            if self._settings['k'] > 0:
                score = self._score_documents(
                    context_filtered, golden_doc_chunks)
                return {f"exact_match@{self._settings['k']}_documents": score}
            score = self._score_documents(context_texts, golden_doc_chunks)
            return {"exact_match_documents": score}

        if self._settings.get("context_type") == "entity":
            if self._settings['k'] > 0:
                return {
                    f"exact_match@{self._settings['k']}_entities": self._score_internal(context_filtered, golden_triples)
                }
            return {
                "exact_match_entities": self._score_internal(context_texts, golden_triples)
            }
        if self._settings['k'] > 0:
            return {
                f"exact_match@{self._settings['k']}_triples": self._score_internal(context_filtered, golden_triples)
            }
        return {
            "exact_match_triples": self._score_internal(context_texts, golden_triples)
        }

    def _score_internal(self, contexts: list[str], golden: Union[str, List[str]]) -> float:
        """
        Calculates the exact match score between predicted contexts and golden (reference) values.

        The score is computed as the ratio of exact matches to total predictions. For entity-type
        contexts, entities are extracted before comparison. Empty/None golden values are handled
        as special cases.

        Parameters:
            contexts (list[str]): List of predicted context strings to evaluate
            golden (Union[str, List[str]]): Golden (reference) value(s) to compare against

        Returns:
            float: Exact match score between 0.0 and 1.0, where:
                - 1.0 means all predictions exactly match the golden value(s)
                - 0.0 means no predictions match the golden value(s)
                - For empty/None golden values:
                    - Returns 1.0 if contexts is also empty/None
                    - Returns 0.0 if contexts contains predictions
        """

        if golden is None or golden == "":
            if contexts is None or len(contexts) == 0:
                return 1
            return 0

        if contexts is None or len(contexts) == 0:
            return 0

        if not isinstance(contexts, list):
            contexts = [contexts]
        if not isinstance(golden, list):
            golden = [golden]

        if self._settings["context_type"] == "entity":
            golden_set = set(self._get_entities(golden))
            contexts = set(self._get_entities(contexts))

        else:
            golden_set = set(golden)

        exact_matches = 0
        for context in contexts:
            if context in golden_set:
                exact_matches += 1

        total_predicted = len(contexts)
        em_score = exact_matches / total_predicted if total_predicted > 0 else 0.0

        return em_score

    def _score_documents(self, contexts: list[str], golden: Union[str, List[str]]) -> float:
        """
        Calculates the exact match score for document chunks.
        For each predicted context, if any golden document chunk is a substring,
        it is considered a match.
        """
        if golden is None or golden == "":
            return 1.0 if not contexts else 0.0

        if not isinstance(golden, list):
            golden = [golden]
        if not isinstance(contexts, list):
            contexts = [contexts]

        matches = 0
        for context in contexts:
            for golden_chunk in golden:
                if golden_chunk in context:
                    matches += 1
                    break
        total = len(contexts)
        return matches / total if total > 0 else 0.0
