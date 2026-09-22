from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.data.models import RetrievalAnswer
from hublink.retrieval.candidate_hub_finder.ann_hub_finder import ANNHubFinder
from hublink.retrieval.candidate_hub_finder.hybrid_search.bm25_fusion_decorator import (
    Bm25FusionDecorator,
)
from hublink.retrieval.candidate_hub_finder.hybrid_search.splade_fusion_decorator import (
    SpladeFusionDecorator,
)
from hublink.retrieval.strategies.base_retrieval_strategy import (
    BaseRetrievalStrategy,
)
from hublink.retrieval.strategies.direct_retrieval_strategy import (
    DirectRetrievalStrategy,
)
from hublink.retrieval.models.processed_question import ProcessedQuestion


def test_retrieval_adds_extracted_keywords_to_answer():
    strategy = object.__new__(DirectRetrievalStrategy)
    processed_question = ProcessedQuestion(
        question="Who published in 2017?",
        keywords=["2017"],
        embeddings=[[1.0, 0.0]],
    )
    strategy._process_question = MagicMock(return_value=processed_question)
    strategy._run_retrieval = MagicMock(
        return_value=RetrievalAnswer(contexts=[])
    )

    result = strategy.retrieval("Who published in 2017?")

    assert result is not None
    assert result.extracted_keywords == ["2017"]


@pytest.mark.parametrize(
    ("use_bm25", "use_splade", "expected_outer", "expected_inner"),
    [
        (False, False, ANNHubFinder, None),
        (True, False, Bm25FusionDecorator, ANNHubFinder),
        (False, True, SpladeFusionDecorator, ANNHubFinder),
        (True, True, SpladeFusionDecorator, Bm25FusionDecorator),
    ],
)
def test_config_selects_available_sparse_channels(
        use_bm25: bool,
        use_splade: bool,
        expected_outer: type,
        expected_inner: type | None):
    settings = SimpleNamespace(
        number_of_hubs=2,
        top_paths_to_keep=2,
        rrf_k=60,
        use_bm25_hybrid_search=use_bm25,
        use_splade_hybrid_search=use_splade,
    )
    retrieval_data = SimpleNamespace(
        graph=MagicMock(),
        llm_adapter=MagicMock(),
        embedding_adapter=MagicMock(),
        settings=settings,
        hub_storage_manager=MagicMock(),
        source_handler=None,
    )
    sparse_storage_manager = MagicMock(has_bm25=True, has_splade=True)

    with patch.object(BaseRetrievalStrategy, "_prepare_utils"):
        strategy = DirectRetrievalStrategy(
            retrieval_data=retrieval_data,
            sparse_storage_manager=sparse_storage_manager,
        )

    assert isinstance(strategy.candidate_hub_finder, expected_outer)
    if expected_inner is not None:
        assert isinstance(
            strategy.candidate_hub_finder.candidate_hub_finder,
            expected_inner,
        )
