from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.data.taxonomy_loader import TaxonomyLoader
from sqa_system.core.language_model.prompt_provider import PromptProvider
from sqa_system.core.data.models.qa_pair import QAPair
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


class Classification(BaseModel):
    """
    A representation of a question classification.
    """
    question: str = Field(...,
                          description="The question to classify.")
    classification: List[str] = Field(...,
                                      description="The chosen classifications.")
    reasoning: str = Field(...,
                           description="The reasoning behind the classifications.")


class QuestionClassifier:
    """
    This class is responsible for classifying questions based on a given
    taxonomy of question and answer types.
    It uses a LLM for this classification task.
    
    Args:
        llm_adapter (LLMAdapter): The LLM adapter to be used for classification.
    """

    def __init__(self, llm_adapter: LLMAdapter):
        self.llm_adapter = llm_adapter
        self.prompt_provider = PromptProvider()
        self.taxonomy_loader = TaxonomyLoader()

    def classify_qa_pair(self, qa_pair: QAPair) -> QAPair:
        """
        Classifies a 'QAPair' by filling in the question and answer types based on the
        question and answer taxonomies by classifying with the LLM.

        Args:
            qa_pair (QAPair): The QAPair to classify.
            
        Returns:
            QAPair: The classified QAPair.
        """
        return qa_pair
        

    def classify_knowledge_representation(self,
                                          question: str,
                                          triples: list[str]) -> Classification | None:
        """
        Classifies based on the triples provided whether the knowledge representation is
        a single fact or a multi fact. 
        As per definition, if there is only one triple, then it is a single fact, otherwise
        it is a multi fact.

        Args:
            triples (list[str]): The triples to classify.
        """
        if not triples:
            return None
        if len(triples) == 1:
            return Classification(
                question=question,
                classification=["Single Fact"],
                reasoning="The question needs only one triple from the KG to be fully answered"
            )
        return Classification(
            question=question,
            classification=["Multi Fact"],
            reasoning="The question requires multiple triples from the KG for a complete answer."
        )

    def classify_answer_contents(self, question: str) -> Classification | None:
        """
        Classifies the answer content of a question.

        Args:
            question (str): The question to classify.
        """

        # Load the taxonomy
        taxonomy_data = self.taxonomy_loader.load_answer_content_taxonomy()
        taxonomy_description = taxonomy_data.model_dump()

        # Prepare the LLM prompt
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "question_classification/answer_content_classification_prompt.yaml")

        # Run the classification
        return self._run_llm_classification(
            prompt_text, taxonomy_description, question)

    def classify_answer_format(self, question: str) -> Classification | None:
        """
        Classifies the answer format of a question.

        Args:
            question (str): The question to classify.
        """

        # Load the taxonomy
        taxonomy_data = self.taxonomy_loader.load_answer_format_taxonomy()
        taxonomy_description = taxonomy_data.model_dump()

        # Prepare the LLM prompt
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "question_classification/answer_format_classification_prompt.yaml")

        # Run the classification
        return self._run_llm_classification(
            prompt_text, taxonomy_description, question)

    def classify_retrieval_operation(self, question: str) -> Classification | None:
        """
        Classifies the retrieval operation of a question.

        Args:
            question (str): The question to classify.
        """

        # Load the taxonomy
        taxonomy_data = self.taxonomy_loader.load_retrieval_operation_taxonomy()
        taxonomy_description = taxonomy_data.model_dump()

        # Prepare the LLM prompt
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "question_classification/retrieval_operation_classification_prompt.yaml")

        # Run the classification
        return self._run_llm_classification(
            prompt_text, taxonomy_description, question)

    def classify_intention_count(self, question: str) -> Classification | None:
        """
        Classifies the question intention of a question.

        Args:
            question (str): The question to classify.
        """

        # Load the taxonomy
        taxonomy_data = self.taxonomy_loader.load_intention_count_taxonomy()
        taxonomy_description = taxonomy_data.model_dump()

        # Prepare the LLM prompt
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "question_classification/intention_count_classification_prompt.yaml")

        # Run the classification
        return self._run_llm_classification(
            prompt_text, taxonomy_description, question)

    def classify_answer_constraints(self, question: str) -> Classification | None:
        """
        Classifies the answer constraints of a question.

        Args:
            question (str): The question to classify.
        """

        # Load the taxonomy
        taxonomy_data = self.taxonomy_loader.load_answer_constraints_taxonomy()
        taxonomy_description = taxonomy_data.model_dump()

        # Prepare the LLM prompt
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "question_classification/answer_constraints_classification_prompt.yaml")

        # Run the classification
        return self._run_llm_classification(
            prompt_text, taxonomy_description, question)

    def classify_question_goal(self, question: str) -> Classification | None:
        """
        Classifies the question goal of a question.

        Args:
            question (str): The question to classify.
        """

        # Load the taxonomy
        taxonomy_data = self.taxonomy_loader.load_question_goal_taxonomy()
        taxonomy_description = taxonomy_data.model_dump()

        # Prepare the LLM prompt
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "question_classification/question_goal_classification_prompt.yaml")

        # Run the classification
        return self._run_llm_classification(
            prompt_text, taxonomy_description, question)

    def classify_content_domain(self, question: str) -> Classification | None:
        """
        Classifies the content domain of a question.

        Args:
            question (str): The question to classify.
        """

        # Load the taxonomy
        taxonomy_data = self.taxonomy_loader.load_content_domain_taxonomy()
        taxonomy_description = taxonomy_data.model_dump()

        # Prepare the LLM prompt
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "question_classification/content_domain_classification_prompt.yaml")

        # Run the classification
        return self._run_llm_classification(
            prompt_text, taxonomy_description, question)

    def classify_answer_credibility(self, question: str) -> Classification | None:
        """
        Classifies the answer credibility of a question.

        Args:
            question (str): The question to classify.
        """

        # Load the taxonomy
        taxonomy_data = self.taxonomy_loader.load_answer_credibility_taxonomy()
        taxonomy_description = taxonomy_data.model_dump()

        # Prepare the LLM prompt
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "question_classification/answer_credibility_classification_prompt.yaml")

        # Run the classification
        return self._run_llm_classification(
            prompt_text, taxonomy_description, question)

    def _run_llm_classification(self,
                                prompt_text: str,
                                taxonomy: str,
                                question: str) -> Classification | None:
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["question", "categories"],
        )

        chain = prompt | self.llm_adapter.llm.with_structured_output(
            schema=Classification)
        try:
            # Do the LLM call
            classification = chain.invoke(
                {"question": question, "categories": taxonomy}
            )
        except Exception as e:
            logger.error(f"Failed to classify question type: {e}")
            return None
        if not isinstance(classification, Classification):
            logger.error("Failed to classify question type")
            return None
        logger.debug(
            f"Classifying Answer Content Type of question: {classification.question}")
        logger.debug(f"Classification: {classification.classification}")
        logger.debug(f"Reasoning: {classification.reasoning}")

        return classification
