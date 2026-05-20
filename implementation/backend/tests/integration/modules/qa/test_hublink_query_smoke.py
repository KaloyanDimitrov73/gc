import asyncio
import os

import pytest

pytestmark = pytest.mark.integration

SMOKE_QUESTION = "Which paper investigates the property robustness?"


def test_hublink_query_smoke(hublink_service):
    answer, nodes, sources = asyncio.run(
        hublink_service.query(
            question=SMOKE_QUESTION,
            retrieval_mode="direct",
            llm_model=os.getenv("TEST_VDL_MODEL", "qwen3:14b"),
            number_of_hubs=10,
            topic_entity_id=None,
        )
    )

    assert isinstance(answer, str)
    assert answer.strip()
    assert isinstance(nodes, list)
    assert isinstance(sources, list)
