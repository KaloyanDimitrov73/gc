"""Shared pytest configuration for all backend test trees."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services and credentials.",
    )


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--run-integration"):
        return False
    return "integration" in collection_path.parts


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="integration test (use --run-integration to include)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

