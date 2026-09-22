from enum import Enum

from experimentation.config.evaluator_config import EvaluatorConfig
from ...evaluation.base.evaluator import Evaluator
from ...evaluation.implementations.ragas_evaluator.ragas_evaluator import RagasEvaluator
from ...evaluation.implementations.hit_at_k_evaluator import HitAtKEvaluator
from ...evaluation.implementations.map_at_k_evaluator import MAPAtKEvaluator
from ...evaluation.implementations.mrr_at_k_evaluator import MRRAtKEvaluator
from ...evaluation.implementations.basic_score_evaluator import BasicScoreEvaluator
from ...evaluation.implementations.bert_score_evaluator import BertScoreEvaluator
from ...evaluation.implementations.exact_match_evaluator import ExactMatchEvaluator
from ...evaluation.implementations.rouge_score_evaluator import RougeScoreEvaluator
from ...evaluation.implementations.bleu_score_evaluator import BleuScoreEvaluator
from ...evaluation.implementations.instruction_following_evaluator import InstructionFollowingEvaluator


class EvaluatorType(Enum):
    """The type of the evaluator."""
    RAGAS = "ragas"
    HITATONE = "hit_at_k"
    MAP = "map_at_k"
    MRR = "mrr_at_k"
    BASIC_SCORE = "basic_score"
    BERT_SCORE = "bert_score"
    EXACT_MATCH = "exact_match"
    ROUGE = "rouge_score"
    BLEU = "bleu_score"
    INSTRUCTION_FOLLOWING = "instruction_following"


class EvaluatorFactory:
    """A factory class for creating evaluators."""

    @staticmethod
    def create(config: EvaluatorConfig) -> Evaluator:
        """
        Creates an evaluator with the given type.
        
        Args:
            config (EvaluatorConfig): The configuration for the evaluator from which the class
                should be created.
        
        Returns:
            Evaluator: The evaluator class that has been created.
        """
        if config.evaluator_type in {EvaluatorType.RAGAS, EvaluatorType.RAGAS.value}:
            return RagasEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.HITATONE, EvaluatorType.HITATONE.value}:
            return HitAtKEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.MAP, EvaluatorType.MAP.value}:
            return MAPAtKEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.MRR, EvaluatorType.MRR.value}:
            return MRRAtKEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.BASIC_SCORE, EvaluatorType.BASIC_SCORE.value}:
            return BasicScoreEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.BERT_SCORE, EvaluatorType.BERT_SCORE.value}:
            return BertScoreEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.EXACT_MATCH, EvaluatorType.EXACT_MATCH.value}:
            return ExactMatchEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.ROUGE, EvaluatorType.ROUGE.value}:
            return RougeScoreEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.BLEU, EvaluatorType.BLEU.value}:
            return BleuScoreEvaluator(config=config)
        if config.evaluator_type in {EvaluatorType.INSTRUCTION_FOLLOWING, EvaluatorType.INSTRUCTION_FOLLOWING.value}:
            return InstructionFollowingEvaluator(config=config)

        raise ValueError(f"Unknown evaluator type: {config.evaluator_type}")

    @classmethod
    def get_evaluator_class(cls, evaluator_type: str) -> type[Evaluator]:
        """
        Returns the class of the evaluator with the specified type.
        
        Args:
            evaluator_type (str): The type of the evaluator as a string.
            
        Returns:
            type[Evaluator]: The class of the evaluator with the specified type.
            
        Raises:
            ValueError: If the evaluator type is not recognized.
        """
        if evaluator_type == EvaluatorType.RAGAS.value:
            return RagasEvaluator
        if evaluator_type == EvaluatorType.HITATONE.value:
            return HitAtKEvaluator
        if evaluator_type == EvaluatorType.MAP.value:
            return MAPAtKEvaluator
        if evaluator_type == EvaluatorType.MRR.value:
            return MRRAtKEvaluator
        if evaluator_type == EvaluatorType.BASIC_SCORE.value:
            return BasicScoreEvaluator
        if evaluator_type == EvaluatorType.BERT_SCORE.value:
            return BertScoreEvaluator
        if evaluator_type == EvaluatorType.EXACT_MATCH.value:
            return ExactMatchEvaluator
        if evaluator_type == EvaluatorType.ROUGE.value:
            return RougeScoreEvaluator
        if evaluator_type == EvaluatorType.BLEU.value:
            return BleuScoreEvaluator
        if evaluator_type == EvaluatorType.INSTRUCTION_FOLLOWING.value:
            return InstructionFollowingEvaluator

        raise ValueError(f"Unknown evaluator type: {evaluator_type}")

    @classmethod
    def get_evaluator_class_by_metric_name(cls, metric_name: str) -> type[Evaluator]:
        """
        Returns the class of the evaluator that calculates the specified metric.
        
        Args:
            metric_name (str): The name of the metric that the returned evaluator 
                calculates.
                
        Returns:
            type[Evaluator]: The class of the evaluator that calculates the 
                specified metric.
        Raises:
            ValueError: If the metric name is not recognized.
        """
        if metric_name in {evaluator.get_metric_names() for evaluator in EvaluatorFactory.get_all_evaluators()}:
            return EvaluatorFactory.get_evaluator_class(metric_name)
        raise ValueError(f"Unknown metric name: {metric_name}")