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


def test_question_processing_parser_uses_last_valid_dictionary():
    output = """
    {"components": ["Example"], "keywords": []}
    Explanation emitted by the model.
    {"components": [" Research Object ", "Technical Debt", "Technical Debt"],
     "keywords": [" 2017 ", "2017", " "]}
    """

    components, keywords = (
        BaseRetrievalStrategy._extract_question_processing(output)
    )

    assert components == ["Research Object", "Technical Debt"]
    assert keywords == ["2017"]


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


@pytest.mark.parametrize("output", [
    "not a dictionary",
    "{'components': ['Technical Debt']}",
    "{'components': 'Technical Debt', 'keywords': []}",
    "{'components': ['Technical Debt'], 'keywords': [2017]}",
])
def test_invalid_question_processing_output_falls_back_to_dense(output: str):
    assert BaseRetrievalStrategy._extract_question_processing(output) == (
        [], []
    )


@pytest.mark.parametrize(
    (
        "extract_components",
        "use_bm25",
        "use_splade",
        "expects_llm_processing",
        "expected_components",
    ),
    [
        (False, False, False, False, []),
        (True, False, False, True, ["Research Object", "Technical Debt"]),
        (False, True, False, False, []),
        (False, False, True, False, []),
    ],
)
def test_question_processing_runs_only_when_component_extraction_is_enabled(
        extract_components: bool,
        use_bm25: bool,
        use_splade: bool,
        expects_llm_processing: bool,
        expected_components: list[str]):
    strategy = object.__new__(DirectRetrievalStrategy)
    strategy.settings = SimpleNamespace(
        extract_question_components=extract_components,
        use_bm25_hybrid_search=use_bm25,
        use_splade_hybrid_search=use_splade,
    )
    strategy._get_question_processing = MagicMock(return_value=(
        ["Research Object", "Technical Debt"],
        ["2017"],
    ))
    strategy.embedding_model = MagicMock()
    strategy.embedding_model.embed_batch.return_value = [[1.0, 0.0]]
    strategy.progress_handler = MagicMock()

    processed = strategy._process_question("question")

    assert processed.components == expected_components
    assert processed.keywords == (["2017"] if expects_llm_processing else [])
    assert strategy._get_question_processing.called is expects_llm_processing
    strategy.embedding_model.embed_batch.assert_called_once_with(
        ["question"] + expected_components
    )


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
