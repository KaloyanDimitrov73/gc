from typing import Any, Tuple
from collections import defaultdict
from enum import Enum
import nltk
from ragas.metrics import (
    FactualCorrectness,
    ContextEntityRecall,
    Faithfulness,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
    SemanticSimilarity,
    NonLLMStringSimilarity
)


class RagasMetrics(Enum):
    """
    This enum stores all metrics from RAGAS that we prepared for the evaluation.
    It is used during the configuration setup to specify which metrics should be used
    for each experiment.
    """
    FACTUAL_CORRECTNESS_F1 = "factual_correctness_f1"
    FACTUAL_CORRECTNESS_PRECISION = "factual_correctness_precision"
    FACTUAL_CORRECTNESS_RECALL = "factual_correctness_recall"
    FAITHFULNESS = "faithfulness"
    LLM_CONTEXT_RECALL = "llm_context_recall"
    LLM_CONTEXT_PRECISION_WITH_REFERENCE = "llm_context_precision_with_reference"
    LLM_CONTEXT_PRECISION_WITHOUT_REFERENCE = "llm_context_precision_without_reference"
    CONTEXT_ENTITY_RECALL = "context_entity_recall"
    RESPONSE_RELEVANCY = "response_relevancy"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    NON_LLM_STRING_SIMILARITY = "non_llm_string_similarity"


class RagasMetricPreparer:
    """
    A specific class that handles the preparation of RAGAS metrics
    """

    # We encountered an issue with RAGAS where if the class is instantiated
    # multiple times, it will not work properly because the examples in prompts
    # are appended each time. This results in overally long prompts which eventually
    # reach the context limit. To solve this, we use a dictionary to store
    # the metrics to avoid re-initializing them multiple times.
    _prepared_metrics = {}

    def prepare_metrics(self, metric_names: list[str]) -> dict:
        """
        The main function that is responsible for the preparation of the metrics.

        Args:
            metric_names (list[str]): The names of the metrics to be prepared.

        Returns:
            dict: A dictionary containing the prepared metrics.
        """

        # Ragas BLEU score needs the nltk "punkt_tab" package to be ready
        # In the current implementation, they are not downloaded it by themselves
        # which results in an error. Therefore we download it here.
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")

        ragas_metrics = {}
        self._prepare_llm_metrics(metric_names, ragas_metrics)
        self._prepare_non_llm_metrics(metric_names, ragas_metrics)

        return ragas_metrics

    def _prepare_llm_metrics(self, metric_names: list[str], ragas_metrics: dict):
        """
        Prepares all LLM based metrics.

        Args:
            metric_names (list[str]): The names of the metrics to be prepared.
            ragas_metrics (dict): The dictionary to store the prepared metrics
                which will be filled during the preparation process.
        """

        for metric in metric_names:
            if metric == RagasMetrics.FACTUAL_CORRECTNESS_F1.value:
                ragas_metrics.update(self._prepare_factual_correctness("f1"))
            elif metric == RagasMetrics.FACTUAL_CORRECTNESS_PRECISION.value:
                ragas_metrics.update(
                    self._prepare_factual_correctness("precision"))
            elif metric == RagasMetrics.FACTUAL_CORRECTNESS_RECALL.value:
                ragas_metrics.update(
                    self._prepare_factual_correctness("recall"))
            elif metric == RagasMetrics.FAITHFULNESS.value:
                ragas_metrics.update(self._prepare_faithfulness())
            elif metric == RagasMetrics.LLM_CONTEXT_RECALL.value:
                ragas_metrics.update(self._prepare_llm_context_recall())
            elif metric == RagasMetrics.LLM_CONTEXT_PRECISION_WITH_REFERENCE.value:
                ragas_metrics.update(self._prepare_llm_context_precision(True))
            elif metric == RagasMetrics.LLM_CONTEXT_PRECISION_WITHOUT_REFERENCE.value:
                ragas_metrics.update(
                    self._prepare_llm_context_precision(False))
            elif metric == RagasMetrics.CONTEXT_ENTITY_RECALL.value:
                ragas_metrics.update(self._prepare_context_entity_recall())
            elif metric == RagasMetrics.RESPONSE_RELEVANCY.value:
                ragas_metrics.update(self._prepare_response_relevancy())
            elif metric == RagasMetrics.SEMANTIC_SIMILARITY.value:
                ragas_metrics.update(self._prepare_semantic_similarity())

    def _prepare_non_llm_metrics(self, metric_names: list[str], ragas_metrics: dict):
        """
        Prepares those metrics that are not LLM based.

        Args:
            metric_names (list[str]): The names of the metrics to be prepared.
            ragas_metrics (dict): The dictionary to store the prepared metrics
                which will be filled during the preparation process.
        """
        for metric in metric_names:
            if metric == RagasMetrics.NON_LLM_STRING_SIMILARITY.value:
                ragas_metrics.update(self._prepare_non_llm_string_similarity())

    def _prepare_factual_correctness(self, mode: str) -> Tuple[str, Any]:
        """Prepares the factual correctness metric with the specified mode."""
        # Ragas implicitly adds the mode of the metric to the name. Because we need
        # the dictionary key to match exactly the output of ragas, we add the mode
        # to the name of the metric.
        name = FactualCorrectness.name + f"(mode={mode})"

        # If the metric is already prepared, we return it from the dictionary
        # to avoid re-initializing it.
        if self._prepared_metrics.get(name) is not None:
            return {name: self._prepared_metrics[name]}

        metric = FactualCorrectness(
            mode=mode,
            atomicity="high",
            coverage="high",
            name=FactualCorrectness.name
        )

        # There is a bug where the examples would be duplicated when different modes
        # are initialized. Therefore we remove the duplicates here
        counter = defaultdict(int)
        deduplicated_examples = []
        for example in metric.claim_decomposition_prompt.examples:
            if not counter[str(example)]:
                deduplicated_examples.append(example)
            counter[str(example)] += 1
        metric.claim_decomposition_prompt.examples = deduplicated_examples

        self._prepared_metrics[name] = metric
        return {name: metric}

    def _prepare_faithfulness(self) -> Tuple[str, Any]:
        """Prepares the faithfulness metric."""
        if self._prepared_metrics.get(Faithfulness.name) is not None:
            return {Faithfulness.name: self._prepared_metrics[Faithfulness.name]}
        metric = Faithfulness()
        self._prepared_metrics[Faithfulness.name] = metric
        return {Faithfulness.name: metric}

    def _prepare_llm_context_recall(self) -> Tuple[str, Any]:
        """Prepares the LLM context recall metric."""
        if self._prepared_metrics.get(LLMContextRecall.name) is not None:
            return {LLMContextRecall.name: self._prepared_metrics[LLMContextRecall.name]}

        name = LLMContextRecall.name
        metric = LLMContextRecall()

        self._prepared_metrics[name] = metric
        return {name: metric}

    def _prepare_llm_context_precision(self, with_reference: bool) -> Tuple[str, Any]:
        """Prepares the LLM context precision metric."""
        if with_reference:
            name = LLMContextPrecisionWithReference.name
            if self._prepared_metrics.get(name) is not None:
                return {LLMContextPrecisionWithReference.name: self._prepared_metrics[LLMContextPrecisionWithReference.name]}

            metric = LLMContextPrecisionWithReference()
            self._prepared_metrics[name] = metric
            return {name: metric}
        name = LLMContextPrecisionWithoutReference.name
        if self._prepared_metrics.get(name) is not None:
            return {name: self._prepared_metrics[name]}
        metric = LLMContextPrecisionWithoutReference()
        self._prepared_metrics[name] = metric
        return {name: metric}

    def _prepare_context_entity_recall(self) -> Tuple[str, Any]:
        """Prepares the context entity recall metric."""
        name = ContextEntityRecall.name
        if self._prepared_metrics.get(name) is not None:
            return {name: self._prepared_metrics[name]}

        self._prepared_metrics[name] = ContextEntityRecall()
        return {name: self._prepared_metrics[name]}

    def _prepare_response_relevancy(self) -> Tuple[str, Any]:
        """Prepares the response relevancy metric."""
        name = ResponseRelevancy.name
        if self._prepared_metrics.get(name) is not None:
            return {ResponseRelevancy.name: self._prepared_metrics[name]}
        self._prepared_metrics[name] = ResponseRelevancy()
        return {name: self._prepared_metrics[name]}

    def _prepare_semantic_similarity(self) -> Tuple[str, Any]:
        """Prepares the semantic similarity metric."""
        name = SemanticSimilarity.name
        if self._prepared_metrics.get(name) is not None:
            return {name: self._prepared_metrics[name]}
        self._prepared_metrics[name] = SemanticSimilarity()
        return {name: self._prepared_metrics[name]}

    def _prepare_non_llm_string_similarity(self) -> Tuple[str, Any]:
        """Prepares the non-LLM string similarity metric."""
        name = NonLLMStringSimilarity.name
        if self._prepared_metrics.get(name) is not None:
            return {NonLLMStringSimilarity.name: self._prepared_metrics[name]}
        self._prepared_metrics[name] = NonLLMStringSimilarity()
        return {name: self._prepared_metrics[name]}
