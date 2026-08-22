from typing import List, Optional, ClassVar
from typing_extensions import override
import weave
from pydantic import model_validator, BaseModel, Field
from langchain_core.prompts import PromptTemplate

from implementation.core import LLMConfig
from language_model import LLMProvider, LLMAdapter
from language_model import EndpointType
from implementation.core import ProgressHandler
from implementation.core import AdditionalConfigParameter
from implementation.language_model import PromptProvider
from implementation.experimentation.evaluation.base.evaluator import Evaluator
from implementation.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_LLM_CONFIG = LLMConfig(
    endpoint=EndpointType.OPENAI.value,
    name_model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=-1,
)


class InstructionFollowingOutput(BaseModel):
    """
    Class that represents the output that the LLM should return.
    """
    reasoning: str = Field(
        ...,
        description="The reasoning on whether the instructions were followed or not."
    )
    follows_instructions: bool = Field(
        ...,
        description="True if the answer followed the instructions, False otherwise."
    )


class InstructionFollowingEvaluator(Evaluator):
    """
    A prompt based LLM evaluator that checks whether the LLM followed the instructions
    that were given in the question.

    It returns 1.0 if the LLM followed the instructions and 0.0 otherwise.
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        AdditionalConfigParameter(
            name="llm_config",
            description="The language model to use for evaluation.",
            param_type=LLMConfig,
            default_value=DEFAULT_LLM_CONFIG
        )
    ]

    _settings: dict = {}
    _initialized: bool = False
    _prompt_provider: PromptProvider = None
    _llm: LLMAdapter = None

    @model_validator(mode="after")
    def initialize_late(self) -> "InstructionFollowingEvaluator":
        """Initialize after model validation."""
        if not self._initialized:
            self._settings = AdditionalConfigParameter.validate_dict(
                self.ADDITIONAL_CONFIG_PARAMS, self.config.additional_params
            )
            self._llm = LLMProvider().get_llm_adapter(
                self._settings["llm_config"])
            self._prompt_provider = PromptProvider()
            self.name = "InstructionFollowingEvaluator"
            self._initialized = True
        return self

    @override
    def get_metric_names(self) -> List[str]:
        return [
            "instruction_following"
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
        Computes the instruction following by using an LLM to evaluate whether the 
        generated answer follows the instructions given in the question.
        """
        ProgressHandler().update_task_by_string_id("evaluating_results")
        logger.debug("Starting Instruction Following evaluation.")

        if self._llm is None or self._prompt_provider is None:
            return {
                "instruction_following": 0.0
            }

        if output is None or "generated_answer" not in output:
            return {
                "instruction_following": 0.0
            }

        generated_answer = output["generated_answer"]
        question = output["initial_question"]
        if (generated_answer is None or question is None or
                golden_answer is None):
            return {
                "instruction_following": 0.0
            }

        return {
            "instruction_following": self._run_instruction_evaluation(
                question=question,
                golden_answer=golden_answer,
                generated_answer=generated_answer
            )
        }

    def _run_instruction_evaluation(self, question: str, golden_answer: str, generated_answer: str):
        """
        Runs the instruction evaluation using the LLM.
        """
        prompt_text, input_variables, _ = self._prompt_provider.get_prompt(
            "instruction_evaluator/instruction_following_evaluation_prompt.yaml")

        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=input_variables,
        )

        retry_count = 3
        while retry_count > 0:
            try:
                chain = prompt | self._llm.llm.with_structured_output(
                    schema=InstructionFollowingOutput)
                response = chain.invoke(
                    {
                        "question": question,
                        "golden_answer": golden_answer,
                        "generated_answer": generated_answer,
                    }
                )
                break
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                retry_count -= 1
                if retry_count == 0:
                    raise e
        if response.follows_instructions:
            return 1.0

        return 0.0
