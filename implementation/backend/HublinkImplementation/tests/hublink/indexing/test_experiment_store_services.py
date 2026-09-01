from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.app.modules.indexing.infrastructure.hublink.hub_store_service import (
    HubStoreService,
)
from hublink.indexing.experiment.sparse_store_service import (
    SparseStoreService,
)


def _settings(**overrides):
    values = {
        "distance_metric": "cosine",
        "embedding_config": object(),
        "diversity_ranking_penalty": 0.05,
        "use_bm25_hybrid_search": True,
        "use_splade_hybrid_search": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@patch(
    "backend.app.modules.indexing.infrastructure.hublink.hub_store_service."
    "HubStorageManager"
)
@patch(
    "backend.app.modules.indexing.infrastructure.hublink.hub_store_service."
    "VectorStoreProvider"
)
@patch(
    "backend.app.modules.indexing.infrastructure.hublink.hub_store_service."
    "HubLinkSettings"
)
@patch(
    "backend.app.modules.indexing.infrastructure.hublink.hub_store_service."
    "get_settings"
)
def test_hub_store_service_creates_only_dense_store(
        get_settings,
        hub_link_settings,
        vector_store_provider,
        hub_storage_manager) -> None:
    config = SimpleNamespace(index_llm_config=object())
    settings = _settings()
    vector_store = object()
    get_settings.return_value = SimpleNamespace(vector_store_backend="chroma")
    hub_link_settings.from_config.return_value = settings
    vector_store_provider.compute_store_name.return_value = "dense-key"
    vector_store_provider.get_vector_store.return_value = vector_store

    service = HubStoreService(config)

    vector_store_provider.compute_store_name.assert_called_once_with(config, settings)
    vector_store_provider.get_vector_store.assert_called_once_with(
        backend="chroma",
        store_name="dense-key",
        distance_metric=settings.distance_metric,
    )
    hub_storage_manager.assert_called_once_with(
        vector_store=vector_store,
        embedding_config=settings.embedding_config,
        diversity_penalty=settings.diversity_ranking_penalty,
    )
    assert service.hub_storage_manager is hub_storage_manager.return_value
    assert not hasattr(service, "sparse_storage_manager")


@patch(
    "hublink.indexing.experiment.sparse_store_service."
    "SparseStorageManager"
)
@patch(
    "hublink.indexing.experiment.sparse_store_service."
    "HubLinkSettings"
)
def test_sparse_store_service_uses_enabled_channels_by_default(
        hub_link_settings,
        sparse_storage_manager) -> None:
    config = Mock()
    hub_link_settings.from_config.return_value = _settings()

    service = SparseStoreService(config)

    sparse_storage_manager.from_config.assert_called_once_with(
        config=config,
        load_bm25=True,
        load_splade=False,
    )
    assert (
        service.sparse_storage_manager
        is sparse_storage_manager.from_config.return_value
    )


@patch(
    "hublink.indexing.experiment.sparse_store_service."
    "SparseStorageManager"
)
@patch(
    "hublink.indexing.experiment.sparse_store_service."
    "HubLinkSettings"
)
def test_sparse_store_service_accepts_channel_loading_overrides(
        hub_link_settings,
        sparse_storage_manager) -> None:
    config = Mock()
    hub_link_settings.from_config.return_value = _settings()

    SparseStoreService(
        config,
        load_bm25=False,
        load_splade=True,
    )

    sparse_storage_manager.from_config.assert_called_once_with(
        config=config,
        load_bm25=False,
        load_splade=True,
    )
