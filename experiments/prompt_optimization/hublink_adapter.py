"""Adapter that evaluates question processing before answer generation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RawRetrievalResult:
    """Ranked retrieval evidence returned before LLM answer generation."""

    source_ids: list[str]
    triples: list[str]


def _deduplicate(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


class HubLinkRawRetrievalAdapter:
    """Run HubLink with externally supplied components and keywords.

    The adapter intentionally stops after HubLink's candidate-path scoring and
    hub pruning. It excludes partial/final answer generation, which isolates the
    question-processing prompt and makes optimizer evaluations substantially
    cheaper.
    """

    def __init__(self, retriever: Any):
        self.retriever = retriever
        if retriever.settings.use_topic_if_given:
            raise ValueError(
                "Prompt optimization currently supports direct retrieval only. "
                "Set use_topic_if_given=false in the experiment config."
            )

    def __deepcopy__(self, memo: dict[int, Any]) -> "HubLinkRawRetrievalAdapter":
        # DSPy deep-copies programs during compilation. The graph and vector
        # stores are shared read-only resources and are not safely deepcopyable.
        del memo
        return self

    @property
    def sparse_routing_enabled(self) -> bool:
        settings = self.retriever.settings
        return bool(
            settings.use_bm25_hybrid_search
            or settings.use_splade_hybrid_search
        )

    def process_with_legacy_prompt(
        self,
        question: str,
    ) -> tuple[list[str], list[str]]:
        """Execute the current YAML question-processing prompt."""

        strategy = self._new_strategy()
        return strategy._get_question_processing(question)  # noqa: SLF001

    def retrieve(
        self,
        question: str,
        components: list[str],
        keywords: list[str],
    ) -> RawRetrievalResult:
        """Retrieve ranked HubLink evidence without generating an answer."""

        from core.data.models import Triple
        from hublink.retrieval.models.processed_question import ProcessedQuestion

        strategy = self._new_strategy()
        embeddings = self.retriever.embedding_model.embed_batch(
            [question, *components]
        )
        if embeddings is None:
            raise RuntimeError("Embedding model returned None.")

        processed_question = ProcessedQuestion(
            question=question,
            components=components,
            keywords=keywords,
            embeddings=embeddings,
        )
        candidate_hubs = strategy._find_candidate_hubs(  # noqa: SLF001
            processed_question
        )
        candidate_hubs = strategy._fill_or_remove_paths(  # noqa: SLF001
            processed_question=processed_question,
            candidate_hubs=candidate_hubs,
            path_threshold=strategy.settings.top_paths_to_keep,
        )
        hubs = strategy._convert_to_hubs(candidate_hubs)  # noqa: SLF001
        ranked_hubs = strategy._prune_hubs(  # noqa: SLF001
            hubs=hubs,
            alpha=strategy.settings.path_weight_alpha,
        )

        source_ids: list[str] = []
        triples: list[str] = []
        for hub in ranked_hubs:
            source_ids.append(
                strategy.answer_generator._get_source_identifier_of_hub_entity(  # noqa: SLF001
                    hub.root_entity
                )
            )
            for path in hub.paths:
                triples.extend(Triple.convert_list_to_string(path.path))

        return RawRetrievalResult(
            source_ids=_deduplicate(source_ids),
            triples=_deduplicate(triples),
        )

    def _new_strategy(self) -> Any:
        from hublink.retrieval.strategies.base_retrieval_strategy import (
            RetrievalStrategyData,
        )
        from hublink.retrieval.strategies.direct_retrieval_strategy import (
            DirectRetrievalStrategy,
        )

        retrieval_data = RetrievalStrategyData(
            graph=self.retriever.graph,
            llm_adapter=self.retriever.query_llm,
            embedding_adapter=self.retriever.embedding_model,
            settings=self.retriever.settings,
            hub_storage_manager=self.retriever.hub_storage_manager,
            source_handler=self.retriever.hub_source_handler,
        )
        strategy_class = DirectRetrievalStrategy
        if self.retriever.settings.use_cross_encoder:
            from hublink.retrieval.strategies.cross_encoder_direct_retrieval_strategy import (
                CrossEncoderDirectRetrievalStrategy,
            )

            strategy_class = CrossEncoderDirectRetrievalStrategy

        return strategy_class(
            retrieval_data=retrieval_data,
            sparse_storage_manager=self.retriever.sparse_storage_manager,
        )


def load_retrieval_adapter(
    pipeline_config_path: Path | str,
) -> tuple[HubLinkRawRetrievalAdapter, Any]:
    """Build the experiment HubLink retriever from an existing pipeline config."""

    # KGRetrievalPipe reuses HubStoreService from the local API package. Installing
    # the complete API package would pull Guardrails, whose Faker constraint
    # conflicts with ORKG's. The service itself needs no Guardrails, so expose the
    # local backend package directly without installing its unrelated API deps.
    repository_root = Path(__file__).resolve().parents[2]
    implementation_root = repository_root / "implementation"
    if str(implementation_root) not in sys.path:
        sys.path.insert(0, str(implementation_root))

    from implementation.config.config_models.pipeline_config import PipelineConfig
    from implementation.config.config_models.retrieval.kg_retrieval_config import (
        KGRetrievalConfig,
    )
    from implementation.pipe.retrieval.implementations.kg_retrieval_pipe import (
        KGRetrievalPipe,
    )

    with Path(pipeline_config_path).open(encoding="utf-8") as config_file:
        pipeline_config = PipelineConfig.model_validate(json.load(config_file))

    retrieval_configs = [
        pipe for pipe in pipeline_config.pipes if isinstance(pipe, KGRetrievalConfig)
    ]
    if len(retrieval_configs) != 1:
        raise ValueError(
            "Pipeline config must contain exactly one KG retrieval pipe; "
            f"found {len(retrieval_configs)}."
        )

    retrieval_config = retrieval_configs[0]
    retrieval_pipe = KGRetrievalPipe(retrieval_config)
    return HubLinkRawRetrievalAdapter(retrieval_pipe.retriever), retrieval_config
