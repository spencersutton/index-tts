import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def setup_deterministic_environment():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_default_device("cpu")

    yield


def pytest_addoption(parser):
    parser.addoption("--regen", action="store_true", default=False, help="Regenerate gold standard artifacts")


@pytest.fixture
def regen(request):
    return request.config.getoption("--regen")


@pytest.fixture(scope="module")
def artifact(request):
    return Path(__file__).parent / "artifacts" / f"{request.module.__name__}.pt"
