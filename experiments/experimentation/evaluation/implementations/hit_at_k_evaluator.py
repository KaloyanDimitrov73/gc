from typing import List, Union, Optional, ClassVar
from typing_extensions import override
import weave
from pydantic import model_validator, PrivateAttr

from core import ProgressHandler
from core import AdditionalConfigParameter, RestrictionType
from experimentation.evaluation.base.evaluator import Evaluator
from core.logging.logging import get_logger

logger = get_logger(__name__)


class HitAtKEvaluator(Evaluator):
    """
    Evaluator that computes the Hit@1 metric by checking whether the first K
    retrieved contexts contain all the golden triples.

    This implementation is based on:
    https://docs.ampligraph.org/en/1.2.0/generated/ampligraph.evaluation.hits_at_n_score.html

    Additionally, only evaluating the triples has one issue. There might be cases where in 
    the graph there is more than one triple that is relevant to the question. In this case,
    we only check whether the retrieval at least found the entity that is relevant to the question.
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

    _settings: dict = PrivateAttr(default_factory=dict)
    _initialized: bool = PrivateAttr(default=False)

    @model_validator(mode='after')
    def initialize_late(self) -> 'HitAtKEvaluator':
        """Initialize after model validation"""
        if not self._initialized:
            self._settings = AdditionalConfigParameter.validate_dict(
                self.ADDITIONAL_CONFIG_PARAMS, self.config.additional_params)
            self.name = "HitAtKEvaluator_" + self._settings["context_type"]
            self._initialized = True
        return self
    
    @override
    def get_metric_names(self) -> List[str]:
        if self._settings.get("context_type") == "entity":
            return [f'hit@{self._settings["k"]}_entities']
        if self._settings.get("context_type") == "document":
            return [f'hit@{self._settings["k"]}_documents']
        return [f'hit@{self._settings["k"]}_triples']

    @weave.op()
    @override
    def score(self,
              output: Optional[dict],
              golden_answer: Optional[str] = None,
              golden_triples: Optional[list[str]] = None,
              golden_doc_chunks: Optional[list[str]] = None) -> dict:
        """
        Computes Hit@k by checking if the generated answer matches any golden answer.

        Args:
            model_output (dict): The model's output
            golden_answer (Union[str, List[str]]): The correct answer(s).
        """
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting HIT score evaluation.")
        contexts = self._get_contexts(output)
        context_texts = self._get_context_texts(contexts)
        if self._settings.get("context_type") == "entity":
            return {
                f'hit@{self._settings["k"]}_entities': self._match(context_texts, golden_triples)
            }
        if self._settings.get("context_type") == "document":
            return {
                f'hit@{self._settings["k"]}_documents': self._match(context_texts, golden_doc_chunks)
            }

        return {
            f'hit@{self._settings["k"]}_triples': self._match(context_texts, golden_triples)
        }

    def _match(self, contexts: list[str], golden: Union[str, List[str]]) -> float:
        """
        Checks if the golden contexts are in the first k retrieved contexts.

        Args:
            contexts (List[str]): The retrieved contexts.
            golden (Union[str, List[str]]): The ground truth.
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
            golden = self._get_entities(golden)
            contexts = self._get_entities(contexts)            

        ranks = []
        for g in golden:
            found_rank = None
            for i, context in enumerate(contexts):
                if g.lower() in context.lower():
                    found_rank = i + 1
                    break

            # If not found, treat rank as 'infinite'
            if found_rank is None:
                found_rank = float('inf')

            ranks.append(found_rank)

        k = self._settings["k"]
        hits = sum(rank <= k for rank in ranks)

        # Hits@k = fraction of goldens found at rank <= k
        return hits / len(golden)
