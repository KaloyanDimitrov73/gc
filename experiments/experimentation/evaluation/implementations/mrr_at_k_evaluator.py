from typing import List, Union, Optional, ClassVar
from typing_extensions import override
import weave
from pydantic import model_validator

from core import ProgressHandler
from core import AdditionalConfigParameter, RestrictionType
from experimentation.evaluation.base.evaluator import Evaluator
from core.logging.logging import get_logger

logger = get_logger(__name__)


class MRRAtKEvaluator(Evaluator):
    """
    Evaluator that computes the MRR@K metric by determining the reciprocal rank 
    of the first relevant triple/entity in the top K retrieved contexts.

    The implementation is based on the implementations of the paper:
    "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries"
    The repo can be found here:
    https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py
    we used the commit: 5fc3983
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        AdditionalConfigParameter(
            name="k",
            description="The number of retrieved contexts to consider for the evaluation.",
            param_type=int,
            param_restriction=RestrictionType.GREATER_THAN_ZERO,
            default_value=1
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

    @model_validator(mode='after')
    def initialize_late(self) -> "MRRAtKEvaluator":
        """Initialize after model validation"""
        if not self._initialized:
            self._settings = AdditionalConfigParameter.validate_dict(
                self.ADDITIONAL_CONFIG_PARAMS, self.config.additional_params
            )
            self.name = "MRRAtKEvaluator_" + self._settings["context_type"]
            self._initialized = True
        return self
    
    @override
    def get_metric_names(self) -> List[str]:
        if self._settings.get("context_type") == "document":
            return [f"mrr@{self._settings['k']}_documents"]
        if self._settings["context_type"] == "entity":
            return [f"mrr@{self._settings['k']}_entities"]
        return [f"mrr@{self._settings['k']}_triples"]

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
        Computes MRR@k by finding the reciprocal rank of the first relevant context.
        """
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting MRR score evaluation.")
        contexts = self._get_contexts(output)
        context_texts = self._get_context_texts(contexts)

        if self._settings.get("context_type") == "document":
            metric_name = f'mrr@{self._settings["k"]}_documents'
            return {metric_name: self._mrr_documents(context_texts, golden_doc_chunks)}

        if self._settings["context_type"] == "entity":
            metric_name = f'mrr@{self._settings["k"]}_entities'
        else:
            metric_name = f'mrr@{self._settings["k"]}_triples'
        return {metric_name: self._mrr(context_texts, golden_triples)}

    def _mrr_documents(self, contexts: List[str], golden: Union[str, List[str]]) -> float:
        """
        Calculates the reciprocal rank at K for document chunks using substring matching.
        """
        if golden is None or golden == "":
            return 1.0 if not contexts else 0.0

        if not isinstance(contexts, list):
            contexts = [contexts]
        if not isinstance(golden, list):
            golden = [golden]

        for rank, context in enumerate(contexts[: self._settings["k"]], start=1):
            for gold_item in golden:
                if gold_item.lower() in context.lower():
                    return 1.0 / rank
        return 0.0

    def _mrr(self, contexts: List[str], golden: Union[str, List[str]]) -> float:
        """
        Calculates the reciprocal rank at K for the provided contexts and golden data.
        """
        if golden is None or golden == "":
            return 1.0 if not contexts else 0.0

        if not isinstance(contexts, list):
            contexts = [contexts]
        if not isinstance(golden, list):
            golden = [golden]

        if self._settings["context_type"] == "entity":
            golden = self._get_entities(golden)

        for rank, context in enumerate(contexts[: self._settings["k"]], start=1):
            for gold_item in golden:
                if gold_item.lower() in context.lower():
                    return 1.0 / rank
        return 0.0
