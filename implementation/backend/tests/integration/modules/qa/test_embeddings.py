"""
Integration tests for embedding adapters.

Each test calls the real adapter endpoint and verifies that the returned
embedding is a non-empty list of floats.  Tests are auto-skipped when the
required service is unavailable or an API key is missing.

Required environment variables (loaded from .env automatically):
  VDL_API_KEY                    — API key for the VDL embedding endpoint
  TEST_VDL_EMBED_MODEL           — VDL model to use (default: nomic-embed-text:v1.5)
  KIT_TOOLBOX_API_KEY            — API key for the KIT KI-Toolbox embedding endpoint
  TEST_KIT_TOOLBOX_EMBED_MODEL   — KIT KI-Toolbox model to use (default: kit.qwen3-embedding-8b)
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.integration

SENTENCE = "The quick brown fox jumps over the lazy dog."


def _require_env(name: str) -> str:
    """Return the value of env var ``name``, or skip the test if it is unset."""
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def _assert_embedding_vector(vector):
    """Assert that ``vector`` is a non-empty list of floats."""
    assert isinstance(vector, list)
    assert vector
    assert all(isinstance(value, float) for value in vector)


def _get_embedding_components():
    """Import embedding classes from sqa_system, skipping the test if unavailable."""
    try:
        from core import EmbeddingConfig
        from language_model.implementations.vdl_embedding_adapter import (
            VDLEmbeddingAdapter,
        )
        from language_model.implementations.kittoolbox_embedding_adapter import (
            KitToolboxEmbeddingAdapter,
        )
    except Exception as exc:
        pytest.skip(f"sqa_system dependencies are unavailable: {exc}")

    return EmbeddingConfig, VDLEmbeddingAdapter, KitToolboxEmbeddingAdapter


@pytest.fixture(scope="module", autouse=True)
def load_env():
    """Load .env from the project root so adapter credentials are available."""
    load_dotenv(Path(__file__).resolve().parents[5] / ".env")


@pytest.fixture(scope="module")
def vdl_adapter():
    """Prepared VDLEmbeddingAdapter; skipped if VDL_API_KEY is missing or the service is unreachable."""
    EmbeddingConfig, VDLEmbeddingAdapter, _ = _get_embedding_components()
    _require_env("VDL_API_KEY")

    config = EmbeddingConfig(
        endpoint="VDL",
        name_model=os.getenv("TEST_VDL_EMBED_MODEL", "nomic-embed-text:v1.5"),
    )
    adapter = VDLEmbeddingAdapter(config)
    try:
        adapter.prepare()
    except Exception as exc:
        pytest.skip(f"VDL embedding adapter is not available: {exc}")
    return adapter


@pytest.fixture(scope="module")
def kit_toolbox_adapter():
    """Prepared KitToolboxEmbeddingAdapter; skipped if KIT_TOOLBOX_API_KEY is missing or the service is unreachable."""
    EmbeddingConfig, _, KitToolboxEmbeddingAdapter = _get_embedding_components()
    _require_env("KIT_TOOLBOX_API_KEY")

    config = EmbeddingConfig(
        endpoint="KitToolbox",
        name_model=os.getenv("TEST_KIT_TOOLBOX_EMBED_MODEL", "kit.qwen3-embedding-8b"),
    )
    adapter = KitToolboxEmbeddingAdapter(config)
    try:
        adapter.prepare()
    except Exception as exc:
        pytest.skip(f"KIT KI-Toolbox embedding adapter is not available: {exc}")
    return adapter


@pytest.mark.parametrize("fixture_name", ["vdl_adapter", "kit_toolbox_adapter"])
def test_embed_returns_numeric_vector(request, fixture_name):
    """Each adapter's embed() call should return a non-empty list of floats."""
    adapter = request.getfixturevalue(fixture_name)
    result = adapter.embed(SENTENCE)

    _assert_embedding_vector(result)
