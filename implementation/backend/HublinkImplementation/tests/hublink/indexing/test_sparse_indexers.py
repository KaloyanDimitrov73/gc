from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from hublink.core.models.hub_path import HubPath
from hublink.core.sparse_index.sparse_storage_manager import (
    SparseStorageManager,
)
from hublink.indexing.experiment.sparse_indexing_service_for_experiment import (
    SparseIndexingServiceForExperiment,
)
from hublink.indexing.sparse_index.bm25_indexer import Bm25Indexer
from hublink.indexing.sparse_index.splade_indexer import SpladeIndexer
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexArtifact,
    SparseIndexDocument,
)


def _document() -> SparseIndexDocument:
    return SparseIndexDocument(
        record_id="hub:path",
        text="path text",
        hub_id="hub",
        path_hash="path",
    )


@patch("hublink.indexing.sparse_index.bm25_indexer.bm25s.BM25")
@patch("hublink.indexing.sparse_index.bm25_indexer.tokenize_for_bm25")
def test_bm25_indexer_builds_artifact_for_storage(
        tokenize_for_bm25,
        bm25_class) -> None:
    storage_manager = MagicMock()
    documents = [_document()]
    tokenize_for_bm25.return_value = [["path", "text"]]
    index = bm25_class.return_value

    result = Bm25Indexer(storage_manager).run_indexing(documents)

    bm25_class.assert_called_once_with(
        method="lucene",
        corpus=[documents[0].metadata],
    )
    index.index.assert_called_once_with(
        [["path", "text"]],
        show_progress=False,
    )
    storage_manager.store_bm25.assert_called_once_with(SparseIndexArtifact(
        index=index,
        records=[documents[0].metadata],
    ))
    assert result is storage_manager.store_bm25.return_value


@patch("hublink.indexing.sparse_index.splade_indexer.get_splade_encoder")
def test_splade_indexer_builds_artifact_for_storage(
        get_splade_encoder) -> None:
    storage_manager = MagicMock()
    documents = [_document()]
    matrix = MagicMock()
    encoder = get_splade_encoder.return_value
    encoder.encode_document.return_value = matrix

    result = SpladeIndexer(
        storage_manager,
        model_name="test-model",
    ).run_indexing(documents)

    get_splade_encoder.assert_called_once_with("test-model")
    assert encoder.encode_document.call_args_list == [
        call(
            ["path text"],
            batch_size=16,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )
    ]
    storage_manager.store_splade.assert_called_once_with(SparseIndexArtifact(
        index=matrix,
        records=[documents[0].metadata],
        metadata={"model_name": "test-model"},
    ))
    assert result is storage_manager.store_splade.return_value


@patch(
    "hublink.indexing.experiment.sparse_indexing_service_for_experiment."
    "HubLinkSettings"
)
def test_sparse_indexing_service_uses_channel_indexers(
        hub_link_settings) -> None:
    config = SimpleNamespace(
        knowledge_graph_config=SimpleNamespace(config_hash="graph"),
        index_llm_config=SimpleNamespace(config_hash="llm"),
    )
    hub_link_settings.from_config.return_value = SimpleNamespace(
        use_bm25_hybrid_search=True,
        use_splade_hybrid_search=True,
    )
    path = HubPath(
        path_text="path text",
        path_hash="path",
        path=[],
    )
    hub_storage_manager = MagicMock()
    hub_storage_manager.get_all_hub_paths.return_value = {"hub": [path]}
    bm25_indexer = MagicMock()
    splade_indexer = MagicMock()
    sparse_storage_manager = MagicMock(index_key="graph_llm")
    service = SparseIndexingServiceForExperiment(
        config=config,
        hub_storage_manager=hub_storage_manager,
        sparse_storage_manager=sparse_storage_manager,
        bm25_indexer=bm25_indexer,
        splade_indexer=splade_indexer,
    )

    document_count = service.run_indexing()

    assert document_count == 1
    assert service.index_key == "graph_llm"
    expected_documents = [_document()]
    bm25_indexer.run_indexing.assert_called_once_with(expected_documents)
    splade_indexer.run_indexing.assert_called_once_with(expected_documents)


def test_sparse_storage_manager_has_no_index_building_api() -> None:
    manager = SparseStorageManager("index-key")

    assert not hasattr(manager, "build_bm25")
    assert not hasattr(manager, "build_splade")
