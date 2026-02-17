# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Add a --run-slow and --run-hugemem options to run cpu and memory
# intensive tests

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--run-hugemem",
        action="store_true",
        default=False,
        help="run memory intensive tests",
    )
    parser.addoption(
        "-R", nargs="?", const='any', default='none',
        help="run tests with online data, requires pytest-remotedata",
        dest="remote_data", choices=['astropy', 'any', 'github', 'none'])


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers",
                            "hugemem: mark test as using a lot of memory")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    run_hugemem = config.getoption("--run-hugemem")

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_hugemem = pytest.mark.skip(reason="need --run-hugemem option to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "hugemem" in item.keywords and not run_hugemem:
            item.add_marker(skip_hugemem)
