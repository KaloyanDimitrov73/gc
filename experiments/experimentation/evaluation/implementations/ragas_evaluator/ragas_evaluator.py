from typing import List, ClassVar, Any, Optional
import threading

import logging
import numpy as np
from openai import APIConnectionError
from pydantic import model_validator
from typing_extensions import override
from langchain_core.embeddings import Embeddings
import weave


from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness
from ragas import evaluate as ragas_evaluate
from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from core import ProgressHandler
from language_model import EndpointType
from language_model.base.llm_adapter import LLMAdapter
from core import (
    AdditionalConfigParameter, LLMConfig, EmbeddingConfig)
from experimentation.evaluation.base.evaluator import Evaluator
from language_model import LLMProvider
from core.logging.logging import get_logger

from .utils.ragas_metric_preparer import RagasMetrics, RagasMetricPreparer


logger = get_logger(__name__)

DEFAULT_LLM_CONFIG = LLMConfig(
    endpoint=EndpointType.OPENAI.value,
    name_model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=-1,
)

DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    endpoint=EndpointType.OPENAI.value,
    name_model="text-embedding-3-small",
)

# Unfortunately as of now RAGAS can not be run using the open source models that
# we are capable to run on our server. The largest size are 32b parameter models
# and through our testing, they were not following the instructions to create
# JSON outputs which is a requirement for RAGAS.
# Also see this issue: https://github.com/explodinggradients/ragas/issues/1090
# and https://github.com/explodinggradients/ragas/issues/1364


class RagasEvaluator(Evaluator):
    """
    Evaluator based on the RAGAS framework that provides several metrics
    for LLM based evaluation. The official implementation is available at
    https://github.com/explodinggradients/ragas/
    """
    _llm_adapter: LLMAdapter = None
    _metric_preparer: RagasMetricPreparer
    _embeddings: Embeddings = None
    _lock: Any = None
    _settings: dict = {}
    _ragas_metrics: dict = {}

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        AdditionalConfigParameter(
            name="llm",
            description="The language model to use for evaluation.",
            param_type=LLMConfig,
            default_value=DEFAULT_LLM_CONFIG
        ),
        AdditionalConfigParameter(
            name="embedding_model",
            description="The embeddings to use for evaluation.",
            param_type=EmbeddingConfig,
            default_value=DEFAULT_EMBEDDING_CONFIG
        ),
        AdditionalConfigParameter(
            name="metrics",
            description="The metrics to use for evaluation.",
            param_type=list[str],
            default_value=[metric.value for metric in RagasMetrics]
        )
    ]

    @model_validator(mode="after")
    def initialize_late(self) -> "RagasEvaluator":
        """Initialize after model validation"""
        if self._lock is None:
            self._lock = threading.Lock()
            self._settings = AdditionalConfigParameter.validate_dict(
                self.ADDITIONAL_CONFIG_PARAMS, self.config.additional_params)

            # Prepare the LLM and Embeddings
            llm_config = self._settings.get("llm", DEFAULT_LLM_CONFIG)
            self._llm_adapter = LLMProvider().get_llm_adapter(llm_config)
            embedding_config = self._settings.get(
                "embedding_model", DEFAULT_EMBEDDING_CONFIG)
            self._embeddings = LLMProvider().get_embeddings(embedding_config).embedding
            self._metric_preparer = RagasMetricPreparer()
            self.name = "RagasEvaluator" + \
                "_".join(self._settings["metrics"])
            self._ragas_metrics = self._metric_preparer.prepare_metrics(
                metric_names=self._settings.get(
                    "metrics", [metric.value for metric in RagasMetrics])
            )
        return self

    @override
    def get_metric_names(self) -> List[str]:
        """
        Returns the names of the metrics that are used for evaluation.
        """
        return list(self._ragas_metrics.keys())

    @weave.op()
    @override
    def summarize(self, score_rows) -> dict:
        pass

    @weave.op()
    @override
    def score(self,
              output: Optional[dict],
              golden_answer: Optional[str] = None,
              golden_triples: Optional[list[str]] = None,
              golden_doc_chunks: Optional[list[str]] = None) -> dict:
        """
        Overrides the score method from weave to score the model output.

        The parameters have to match the entries to the given input
        dataset to be fetched sucessfully.
        """
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting RAGAS score evaluation.")
        if golden_answer is None or output is None:
            return {}
        # Pylint throws a false positive error here
        # the lock object is a viable context manager
        # and the code works at runtime.
        # pylint: disable=not-context-manager
        with self._lock:
            return self._score_internal(output, golden_answer, golden_triples)

    def _score_internal(self,
                        model_output: dict,
                        golden_answer: str,
                        golden_triples: Optional[list[str]] = None) -> Any:
        """
        Scores the model output using the Ragas library.

        Args:
            model_output (dict): The model output which is a dictionary of the PipeIOData
                object that has been passed through the pipeline.
            golden_answer (str): The golden answer that is expected.
            golden_triples (Optional[list[str]]): The golden triples that are expected 
                in the context.
        """

        context = self._get_contexts(model_output)

        context_texts = []
        for ctx in context:
            context_texts.append(ctx.text)

        if not context_texts:
            context_texts = self._get_context_texts(context)

        if context is None or len(context) == 0:
            logger.debug("The pipeline didn't return any output.")
            empty_metrics = {}
            for metric in self._ragas_metrics:
                empty_metrics[metric] = 0.0
            return empty_metrics

        # RAGAS needs a special dataset format to evaluate the model
        # therefore we create a qa-dataset with the given model output
        # and convert the dataset to the ragas dataset format needed.
        dataset_dict = {
            "question": model_output["initial_question"],
            "generated_answer": self._clean_answer(model_output["generated_answer"]),
            "golden_triples": golden_triples if golden_triples is not None else [],
            "retrieved_contexts": context_texts,
            "golden_answer": golden_answer
        }
        logger.debug(f"Ragas evaluation dataset: {dataset_dict}")

        dataset = self._convert_to_ragas_dataset(dataset_dict)
        return self._run_ragas_evaluation(dataset, self._ragas_metrics)

    def _run_ragas_evaluation(self,
                              dataset: EvaluationDataset,
                              ragas_metrics: dict) -> dict:
        retry_count = 8
        success = False

        if not ragas_metrics:
            logger.debug("No metrics to evaluate.")
            return {}

        to_eval = dict(ragas_metrics)

        # Create a dictionary to store the metrics
        metrics_answer = {}
        for metric in to_eval:
            metrics_answer[metric] = 0.0
        # If we have no context, we can not have a hallucination. Therefore
        # Faithfulness should be 1
        if Faithfulness.name in to_eval:
            metrics_answer[Faithfulness.name] = 1.0

        while retry_count > 0 and not success:
            try:
                # Prepare the llm
                if self._llm_adapter is None or not isinstance(self._llm_adapter, LLMAdapter):
                    raise ValueError("LLM is not properly initialized")

                run_config = RunConfig(
                    max_workers=4,
                    timeout=600,
                    max_retries=10
                )

                # Disable logging from ragas
                self._prepare_logging()
                evaluation = ragas_evaluate(dataset=dataset,
                                            metrics=list(
                                                to_eval.values()),
                                            raise_exceptions=False,
                                            run_config=run_config,
                                            llm=LangchainLLMWrapper(
                                                self._llm_adapter.llm),
                                            embeddings=LangchainEmbeddingsWrapper(self._embeddings))
                eval_df = evaluation.to_pandas()
                for _, row in eval_df.iterrows():
                    # Create a list of metrics to process
                    metrics_to_process = list(to_eval.keys())

                    # get the metrics from the evaluation
                    for metric in metrics_to_process:
                        value = row[metric]
                        # Here we check if any of the metrics are NaN which indicates
                        # an issue in the calculation. This frequently happens because
                        # of api connection issues. If retried, this should be solved
                        if np.isnan(value):
                            # here we remove the metric from the dictionary to try again
                            logger.warning(
                                "Ragas evaluation for %s failed because of a NaN value. Retrying %s ",
                                metric, retry_count)
                            continue
                        # We also confirm the metric is encoded as float and because ragas
                        # returns values with to many decimal points, we round the value
                        # to 2 decimal points.
                        metrics_answer[metric] = round(float(value), 2)
                        to_eval.pop(metric)
                if len(to_eval) == 0:
                    success = True
                retry_count -= 1
            except APIConnectionError:
                logger.warning(
                    "Ragas evaluation failed because of an API connection error. Retrying %s ",
                    retry_count)
                retry_count -= 1
            except Exception as e:
                logger.error(
                    "Ragas evaluation failed because of an unexpected error: %s. Retrying %s", str(e), retry_count)
                retry_count -= 1
        if not success:
            logger.error(
                "Ragas evaluation could not complete because of errors.")
            logger.debug("Ragas evaluation: %s", str(metrics_answer))
            return metrics_answer
        logger.debug("Ragas evaluation: %s", str(metrics_answer))
        return metrics_answer

    def _convert_to_ragas_dataset(self, dataset_dict: dict) -> EvaluationDataset:
        """
        Ragas uses a special object for datasets. This function converts
        our given data dictionary to this object.
        """

        try:
            # Handle contexts, ensuring it's always a list of strings
            # This is needed because sometimes the retrievers return None values
            # Ragas cant handle that.
            contexts = dataset_dict["retrieved_contexts"]
            if contexts is None:
                contexts = []
            elif isinstance(contexts, str):
                contexts = [contexts]

            contexts = [str(c) if c is not None else "" for c in contexts]
            sample = SingleTurnSample(
                user_input=dataset_dict["question"],
                retrieved_contexts=contexts,
                response=dataset_dict["generated_answer"],
                reference=dataset_dict["golden_answer"],
                reference_contexts=dataset_dict["golden_triples"]
            )
        except AttributeError as e:
            raise AttributeError(
                f"Dict is missing one or more attributes: {dataset_dict}") from e

        ragas_dataset = EvaluationDataset(samples=[sample])

        return ragas_dataset

    def _prepare_logging(self):
        logging.getLogger("ragas.prompt.pydantic_prompt").setLevel(
            logging.CRITICAL)
