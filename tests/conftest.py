import pytest


def pytest_addoption(parser):
    parser.addoption("--run-data-heavy", action="store_true", default=False,
                     help="Run integration demos that read complete local EEG recordings and .pt weights")


def pytest_configure(config):
    config.addinivalue_line("markers", "data_heavy: requires local EEG recordings and trained .pt weights")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-data-heavy"):
        skip = pytest.mark.skip(reason="Requires full local EEG recording and .pt weights; opt in with --run-data-heavy")
        for item in items:
            if "data_heavy" in item.keywords:
                item.add_marker(skip)
