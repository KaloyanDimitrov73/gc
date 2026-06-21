from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import List, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core.logging.logging import get_logger
from core import Triple
from language_model.base.llm_adapter import LLMAdapter
from language_model import PromptProvider
from ..base.knowledge_graph import KnowledgeGraph

logger = get_logger(__name__)


def _llm_call_config(call_type: str) -> dict:
    """Build LangChain run config used by the shared LLM wrapper logger."""
    return {
        "metadata": {"llm_call_type": call_type},
        "tags": [f"llm_call_type:{call_type}"]
    }


class GraphConverter:
    """
    This class is responsible for converting subgraphs and paths into text representations.

    Args:
        llm_adapter (LLMAdapter): The LLM adapter that is used to convert the subgraph and path to text.
        graph (KnowledgeGraph, optional): The knowledge graph to be used. Defaults to None.
    """

    def __init__(self, llm_adapter: LLMAdapter, graph: KnowledgeGraph = None):
        self.graph = graph
        self.llm_adapter = llm_adapter
        self.prompt_provider = PromptProvider()
        self._load_prompts()

    def _load_prompts(self):
        """
        Loads the conversion prompts for subgraphs and paths from the prompt provider.
        """
        self.subgraph_template, self.subgraph_inputs, self.subgraph_partials = (
            self.prompt_provider.get_prompt("conversion/subgraph_to_text.yaml"))
        self.path_template, self.path_inputs, self.path_partials = (
            self.prompt_provider.get_prompt("conversion/path_to_text.yaml"))

    def subgraph_to_description(self, subgraph: List[Triple], llm_adapter: LLMAdapter) -> str:
        """
        Converts a subgraph to a text representation using an LLM.

        Args:
            subgraph (List[Triple]): A list of Triple objects representing the subgraph.
            llm_adapter (LLMAdapter): The LLM adapter that is used to convert the subgraph to text.

        Returns:
            str: A string representation of the subgraph, where the triples of the graph have been
                converted to a natural language description using a language model.
        """
        subgraph_text = self.convert_subgraph_for_llm(subgraph)
        parser = StrOutputParser()

        prompt = PromptTemplate(
            template=self.subgraph_template,
            input_variables=self.subgraph_inputs
        )

        llm_runnable = llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        chain = prompt | llm_runnable | parser

        response = chain.invoke(
            {"subgraph": subgraph_text},
            config=_llm_call_config("graph_converter.subgraph_to_description"))
        return response

    def convert_subgraph_for_llm(self, subgraph: List[Triple]) -> str:
        """
        Converts a subgraph of relations into a string format suitable for use with
        language models (LLMs).

        Args:
            subgraph (List[Relation]): A list of Relation objects representing the subgraph.

        Returns:
            str: A string representation of the subgraph, where each relation is formatted
                 as a triple (head_entity_name, relation_desc, tail_entity_name) and each
                 triple is separated by a newline.
        """
        triples = []
        for relation in subgraph:
            if (relation.entity_subject is None or
                relation.entity_subject.uid is None or
                relation.entity_object is None or
                    relation.entity_object.uid is None):
                continue

            head_entity_name = relation.entity_subject.text
            tail_entity_name = relation.entity_object.text

            description = relation.predicate if relation.predicate else ""
            if description == "":
                continue
            if not description:
                continue
            triple = f"({head_entity_name}, {description}, {tail_entity_name}) "
            triples.append(triple)
        return "\n".join(triples)

    def path_to_text(self, path: List[Triple]) -> str:
        """
        Converts a path to a text representation using an LLM.

        Args:
            path (List[Triple]): A list of Triple objects representing the path.

        Returns:
            str: A string representation of the path, where the triples of the graph have been
                converted to a natural language description using a language model.
        """
        results = self.paths_to_text_batch([path])
        return results[0]

    def paths_to_text_batch(
            self,
            paths: List[List[Triple]],
            max_concurrency: int = 4,
            executor: Optional[ThreadPoolExecutor] = None) -> List[str]:
        """
        Converts multiple paths to text representations in parallel.

        Each path's chain.invoke() runs in a thread. When an external `executor`
        is provided (e.g. a shared one from HubBuilder) it is reused directly,
        capping global OpenAI concurrency across all hubs. Otherwise a local
        ThreadPoolExecutor with `max_concurrency` workers is created.

        Rate-limit responses (HTTP 429) from the LLM API are retried with
        exponential backoff (up to 8 attempts, capped at 60 s per wait).

        Args:
            paths (List[List[Triple]]): The paths to convert.
            max_concurrency (int): Workers for a locally-created pool (ignored
                when `executor` is passed in).
            executor (ThreadPoolExecutor, optional): Shared thread-pool to use.

        Returns:
            List[str]: Text representations in the same order as the input paths.
        """
        logger.debug(
            "Converting %d paths to text in parallel using LLM: %s",
            len(paths), self.llm_adapter.llm_config.name_model)

        results = [""] * len(paths)

        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=self.path_template,
            input_variables=self.path_inputs
        )
        llm_runnable = self.llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        chain = prompt | llm_runnable | parser

        max_retries = 8

        def _convert(idx: int, path: List[Triple]) -> tuple[int, str]:
            if not path:
                return idx, ""
            path_text = self.convert_subgraph_for_llm(path)
            for attempt in range(max_retries):
                try:
                    response = chain.invoke(
                        {"path": path_text},
                        config=_llm_call_config("graph_converter.path_to_text")
                    )
                    return idx, response
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = (
                        "rate limit" in err_str
                        or "429" in err_str
                        or "too many requests" in err_str
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        wait = min(2 ** attempt, 60)
                        logger.warning(
                            "Rate limit hit for path %d, retrying in %ds "
                            "(attempt %d/%d)",
                            idx, wait, attempt + 1, max_retries
                        )
                        time.sleep(wait)
                    else:
                        logger.error("Error converting path %d to text: %s", idx, e)
                        raise

        def _submit_all(pool: ThreadPoolExecutor):
            total = len(paths)
            completed = 0
            lock = threading.Lock()

            futures = {
                pool.submit(_convert, idx, path): idx
                for idx, path in enumerate(paths)
            }
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                with lock:
                    completed += 1
                    logger.info(
                        "Path to text: %d/%d converted (path index %d)",
                        completed,
                        total,
                        idx,
                    )

        if executor is not None:
            _submit_all(executor)
        else:
            with ThreadPoolExecutor(max_workers=max_concurrency) as local_pool:
                _submit_all(local_pool)

        return results
