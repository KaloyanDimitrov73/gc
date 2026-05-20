import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="module", autouse=True)
def load_env():
    load_dotenv(Path(__file__).resolve().parents[5] / ".env")


@pytest.fixture(scope="module")
def hublink_service():
    pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required for HubLink integration tests"
    )

    required_env = ["VDL_API_KEY", "OPENAI_API_KEY"]
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")

    from backend.app.modules.qa.infrastructure.hublink.hublink_service import (
        HubLinkService,
    )

    service = HubLinkService()
    if not service.is_available():
        pytest.skip("HubLink service is not available in this environment")

    return service
