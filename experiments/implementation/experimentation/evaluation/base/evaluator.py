from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Optional, List, ClassVar
import re
from weave.flow.scorer import Scorer, auto_summarize
import weave
from implementation.models.qa_dataset import QADataset
from implementation.core import Context, Triple, ContextType
from implementation.core import AdditionalConfigParameter
from implementation.config.models.evaluator_config import EvaluatorConfig

from implementation.core.logging import get_logger

logger = get_logger(__name__)


class Evaluator(Scorer, ABC):
    """
    A abstract class for evaluators that evaluate a 
    retrieval pipeline with a question-answer dataset.

    It is implemented as a subclass of Scorer from weave.
    The implementation follows the guide from the weave wiki:
    https://weave-docs.wandb.ai/tutorial-rag which has been 
    last accessed on 25-11-2024.
    """

    config: EvaluatorConfig
    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = []

    @override
    @abstractmethod
    @weave.op()
    # pylint throws a false positive error here. The implementation of
    # the score method follows the guide from the wiki of weave.
    # pylint: disable=arguments-differ
    def score(self,
              output: Optional[dict],
              golden_answer: Optional[str] = None,
              golden_triples: Optional[list[str]] = None,
              golden_doc_chunks: Optional[list[str]] = None) -> dict:
        """
        Overrides the score method from weave to score the model output.

        The parameters have to match the entries to the given input
        dataset to be fetched sucessfully.

        Args:
            output (dict): The output of the RAG pipeline filles with the retrieved
                context and the generated answer.
            golden_answer (str): The golden answer to the question as defined in the
                QA dataset.
            golden_triples (list[str]): The golden triples to the question as defined
                in the QA dataset.
            golden_doc_chunks (list[str]): The golden document chunks to the question
                as defined in the QA dataset.

        Returns:
            dict: The scores for each of the evaluators that have been applied.
        """

    @override
    @weave.op()
    def summarize(self, score_rows) -> dict:
        """
        Overrides the summarize method from weave to also 
        append the row_scores to the summarization.

        Args:
            score_rows (list[dict]): The rows to be summarized containing the metric
                values for each question asked.
        """
        summarization = auto_summarize(score_rows)
        if not summarization:
            return {}

        row_scores = []
        for row in score_rows:
            row_dict = {}
            for key, value in row.items():
                row_dict[key] = value
            row_scores.append(row_dict)
        summarization["row_scores"] = row_scores

        return summarization

    @classmethod
    def create_config(cls,
                      evaluator_type: str,
                      name: Optional[str] = None,
                      **kwargs) -> EvaluatorConfig:
        """
        Creates a EvaluatorConfig object with the specified parameters.

        Args:
            evaluator_type (str): The type of the evaluator that should be applied
                for the calculation of the scores.
            name (str, optional): The name of the evaluator. If not provided, the
                name will be generated automatically.
            **kwargs: Additional parameters for the evaluator. The parameters
                have to be defined in the class variable ADDITIONAL_CONFIG_PARAMS
                of the subclass of Evaluator.

        Returns:
            EvaluatorConfig: The configuration object for the evaluator.
        """
        # Validate other config params
        AdditionalConfigParameter.validate_config_params(additional_params=cls.ADDITIONAL_CONFIG_PARAMS,
                                                         **kwargs)

        # Prepare the name if provided
        if name:
            name = EvaluatorConfig.prepare_name_for_config(name)
            return EvaluatorConfig(
                evaluator_type=evaluator_type,
                name=name,
                additional_params=kwargs
            )

        # Return the config without a name
        return EvaluatorConfig(
            evaluator_type=evaluator_type,
            additional_params=kwargs
        )

    def get_metric_names(self) -> List[str]:
        """
        Returns the names of the metrics that are used by the evaluator.

        Returns:
            list[str]: The names of the metrics.
        """
        raise NotImplementedError(
            "The method has to be implemented in the subclass of Evaluator.")

    def validate_qa_dataset(self, dataset: QADataset):
        """
        Validates the given dataset.

        Args:
            dataset (QADataset): The dataset to validate.
        """
        for qa_pair in dataset.get_all_entries():
            if not all([qa_pair.question,
                        qa_pair.golden_answer]):
                raise ValueError(
                    f"QAPair is missing one or more required attributes (question, generated_answer, golden_answer): {qa_pair}")

    def _get_contexts(self, model_output: dict) -> List[Context]:
        """
        Extracts the retrieved contexts from the model output.

        Args:
            model_output (dict): The model output.

        Returns:
            list[Context]: The contexts.
        """
        context = []
        if model_output is None:
            return []
        if "retrieved_context" in model_output:
            if not isinstance(model_output["retrieved_context"], list):
                return [Context(
                    text=model_output["retrieved_context"],
                    context_type=ContextType.KG)]
            try:
                for element in model_output["retrieved_context"]:
                    context.append(Context.model_validate(element))
                return context
            except ValueError:
                logger.debug(
                    "Error parsing 'retrieved_context' using text directly")

            for element in model_output["retrieved_context"]:
                triple = Triple.from_string(element)
                if triple is None:
                    break
                context.append(
                    Context(text=element, context_type=ContextType.KG))

            if len(context) == 0:
                for element in model_output["retrieved_context"]:
                    context.append(
                        Context(text=element, context_type=ContextType.DOC))

        return context

    def _get_context_texts(self, contexts: List[Context]) -> List[str]:
        """
        Extracts for each context the text and returns a list of all texts.

        Args:
            contexts (list[Context]): The contexts to extract the text from.

        Returns:
            list[str]: The texts of the contexts.
        """
        context_texts = []
        for context in contexts:
            context_texts.append(context.text)
        return context_texts

    def _get_entities(self, triples_as_string: List) -> List[str]:
        """
        Extracts all the entities from the given triples and makes sure
        that there are no duplicates.

        Args:
            triples_as_string (list): The triples to extract the entities from.

        Returns:
            list[str]: The entities of the triples.
        """
        entities = set()
        for triple in triples_as_string:
            if not isinstance(triple, str):
                raise ValueError(f"Received non-string triple: {triple}")
            converted_triple = Triple.from_string(triple)
            if converted_triple is None:
                logger.warning("Invalid triple: %s", triple)
                entities.add(triple)
            else:
                entities.add(converted_triple.entity_subject.text)
                entities.add(converted_triple.entity_object.text)

        return list(entities)

    def _clean_answer(self, answer: str) -> str:
        """
        Our HubLink retriever adds source information at the end of each answer e.g.:
        "The author of the paper is John Doe [1].

        [1] Paper Title, DOI: 10.1234/5678"

        This introduces noise in the evaluation and skews the results. For example, because 
        this citation is not part of the golden answer, the BERTScore will be lower than expected.
        Another example is the Faithfulness metric. Because the title or doi does not have to be
        in the golden triples, the metric will thing that HubLink hallucinates the information.

        To avoid this, we remove the list of citations from the answer with this function.
        """
        # Remove the citation part from the answer
        citation_pattern = r"\n\[\d+\][\s\S]*$"
        cleaned_answer = re.sub(citation_pattern, "", answer)
        return cleaned_answer
