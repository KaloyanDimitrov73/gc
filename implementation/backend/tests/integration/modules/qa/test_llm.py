"""
Integration tests for LLM adapters (VDL, OpenAI).

Each test calls the real adapter endpoint and verifies that the response
contains non-empty text content.  Tests are auto-skipped when the required
service is unreachable or an API key is missing.

Required environment variables (loaded from .env automatically):
  VDL_API_KEY         — API key for the VDL LLM endpoint
  TEST_VDL_MODEL      — VDL model to use (default: gpt-oss:20b)
  OPENAI_API_KEY      — API key for the OpenAI endpoint
  TEST_OPENAI_MODEL   — OpenAI model to use (default: o3-mini)
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.integration

PROMPT = "What is 2 + 2? Answer with just the number."


def _require_env(name: str) -> str:
    """Return the value of env var ``name``, or skip the test if it is unset."""
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def _get_llm_components():
    """Import LLM adapter classes from sqa_system, skipping the test if unavailable."""
    try:
        from language_model.config.llm_config import LLMConfig
        from language_model.implementations import OpenAiLLMAdapter
        from language_model.implementations.vdl_llm_adapter import VDLLLMAdapter
    except Exception as exc:
        pytest.skip(f"sqa_system dependencies are unavailable: {exc}")

    return LLMConfig, VDLLLMAdapter, OpenAiLLMAdapter


@pytest.fixture(scope="module", autouse=True)
def load_env():
    """Load .env from the project root so adapter credentials are available."""
    load_dotenv(Path(__file__).resolve().parents[4] / ".env")


@pytest.fixture(scope="module")
def vdl_llm():
    """Prepared VDLLLMAdapter; skipped if VDL_API_KEY is missing or the service is unreachable."""
    LLMConfig, VDLLLMAdapter, _ = _get_llm_components()
    _require_env("VDL_API_KEY")

    config = LLMConfig(
        endpoint="VDL",
        name_model=os.getenv("TEST_VDL_MODEL", "gpt-oss:20b"),
        temperature=0.0,
        max_tokens=64,
    )
    adapter = VDLLLMAdapter(config)
    try:
        adapter.prepare()
        # Probe call to catch auth errors (401) before the test body runs
        adapter.generate("ping")
    except Exception as exc:
        err = str(exc).lower()
        if "401" in err or "session" in err or "token" in err or "unauthorized" in err:
            pytest.fail(
                f"VDL auth failed — key is invalid or expired. "
                f"Generate a permanent API key at "
                f"https://chat.vdl.sdq.kastel.kit.edu (Settings → Account → API Keys).\n"
                f"Original error: {exc}"
            )
        pytest.skip(f"VDL adapter is not available: {exc}")
    return adapter


@pytest.fixture(scope="module")
def openai_llm():
    """Prepared OpenAiLLMAdapter; skipped if OPENAI_API_KEY is missing or the service is unreachable."""
    LLMConfig, _, OpenAiLLMAdapter = _get_llm_components()
    _require_env("OPENAI_API_KEY")

    config = LLMConfig(
        additional_params={},
        endpoint="OpenAI",
        name_model=os.getenv("TEST_OPENAI_MODEL", "o3-mini"),
        temperature=None,
        max_tokens=-1,
        reasoning_effort="low",
    )
    adapter = OpenAiLLMAdapter(config)
    try:
        adapter.prepare()
    except Exception as exc:
        pytest.skip(f"OpenAI adapter is not available: {exc}")
    return adapter


@pytest.mark.parametrize("fixture_name", ["vdl_llm", "openai_llm"])
def test_generate_returns_non_empty_content(request, fixture_name):
    """Each adapter's generate() call should return a response with non-empty string content."""
    adapter = request.getfixturevalue(fixture_name)
    result = adapter.generate(PROMPT)
    content = getattr(result, "content", str(result))

    assert isinstance(content, str)
    assert content.strip()
