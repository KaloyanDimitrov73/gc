"""
Root-level pytest configuration for the backend test suite.

Controls whether integration tests (those under the ``integration/`` directory
or marked with ``@pytest.mark.integration``) are collected and run.

By default integration tests are skipped so that a plain ``pytest`` run only
executes fast, dependency-free unit tests.  Pass ``--run-integration`` on the
command line to include them::

    pytest --run-integration

Hooks
-----
pytest_addoption
    Registers the ``--run-integration`` CLI flag.
pytest_ignore_collect
    Prevents pytest from even collecting files inside ``integration/``
    unless the flag is set.
pytest_collection_modifyitems
    Attaches a skip marker to any already-collected item marked
    ``integration`` when the flag is not set (belt-and-suspenders guard).
"""
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
