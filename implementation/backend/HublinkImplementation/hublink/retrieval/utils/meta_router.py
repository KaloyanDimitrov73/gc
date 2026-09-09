import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core.progress.progress_handler import ProgressHandler
from language_model import LLMAdapter, PromptProvider


# NOTE: adjust these imports to match your actual project structure --
# they mirror the ones implicitly used in generate_instant_response().

# The router only ever needs to emit a single word (see the prompt template),
# so we cap the generation tightly. This keeps latency and cost minimal and
# also acts as a cheap guard against the model rambling.
_ROUTER_MAX_TOKENS = 5

log_file = Path(__file__).parent / "meta_router.log"

routing_logger = logging.getLogger("meta_router")
routing_logger.setLevel(logging.INFO)

handler = logging.FileHandler(log_file, encoding="utf-8")
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

routing_logger.addHandler(handler)


class RouteDecision(str, Enum):
    """The three possible ways of handling a user message."""

    GENERAL = "general"          # answer directly, no grounding needed
    CHAT_HISTORY = "chat_history"  # answerable purely from prior conversation turns
    RETRIEVAL = "retrieval"      # needs the retrieval pipeline


class MetaRouter:
    """
    Decides, based on the current user question and the conversation history,
    whether the answer should be:

      - GENERAL: a direct/instant answer for chit-chat, meta questions about the
        assistant, or questions clearly out of the knowledge base's scope.
      - CHAT_HISTORY: a question that can be fully answered by re-using,
        re-sorting, filtering, or aggregating information that is already
        present in the conversation history (no new facts needed).
      - RETRIEVAL: a question that needs facts not (yet) present in the
        conversation history and therefore requires the retrieval pipeline,
        typically using the history to resolve references like "these papers".

    This router does not itself answer chat_history questions -- it only
    classifies. The caller is expected to dispatch to the appropriate
    downstream component (e.g. an "answer-from-history" generator or the
    retrieval pipeline) based on the returned RouteDecision.
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt_provider = PromptProvider()

    def check_route(
        self,
        question: str,
        history_text: Optional[str] = None,
    ) -> RouteDecision:
        """
        Classifies the incoming question into one of the three routes.

        Args:
            question (str): The current user question.
            history_text (Optional[str]): The prior conversation turns,
                formatted as text (same format you already pass into
                generate_instant_response).

        Returns:
            RouteDecision: One of GENERAL, CHAT_HISTORY, RETRIEVAL.
                Falls back to RETRIEVAL on any parsing/classification failure,
                since retrieval is the safest default (worst case: an
                unnecessary but correct retrieval call).
        """

        _ph = ProgressHandler()
        _ph.add_task(
            string_id="meta_routing",
            description="Deciding how to answer the question",
            total=1,
            reset=True,
        )

        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "novel_retriever/meta_router_prompt.yaml"
        )

        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["question", "history"],
        )

        chain = prompt | self.llm.llm | parser
        raw_response = chain.invoke(
            {
                "question": question,
                "history": history_text or "(no prior conversation)",
            },
            config=LLMAdapter.llm_call_config("hublink.meta_routing"),
        )

        _ph.finish_by_string_id("meta_routing")

        route_value = raw_response.strip().lower()


        try:
            decision = RouteDecision(route_value)
        except ValueError:
            routing_logger.warning(
                "MetaRouter: unparseable route '%s' returned by LLM, "
                "defaulting to RETRIEVAL", route_value,
            )
            return RouteDecision.RETRIEVAL

        routing_logger.info(
            "----------------- TEST 2 -------------------"
        )
        routing_logger.info(
            "QUESTION: %s | ROUTE: %s \n ######### HISTORY: %s",
            question.replace("\n", " "),
            decision.value,
            history_text
        )

        routing_logger.info("")
        return decision