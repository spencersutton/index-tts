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


@pytest.fixture
def artifact(request):
    artifacts_dir = Path(__file__).parent / "artifacts"
    module_name = Path(request.fspath).stem
    function_name = request.node.name

    # Prefer unique, module-scoped artifact names to avoid collisions between
    # identically named tests that live in different files (e.g., multiple
    # `test_stable_result` functions).
    specific_artifact = artifacts_dir / f"{module_name}__{function_name}.pt"
    module_artifact = artifacts_dir / f"{module_name}.pt"
    function_artifact = artifacts_dir / f"{function_name}.pt"

    if specific_artifact.exists():
        return specific_artifact

    # For common test names like `test_stable_result`, prefer the module-scoped
    # artifact to prevent cross-module clashes. Fall back to a unique name if no
    # module artifact is available yet.
    if function_name == "test_stable_result":
        return module_artifact if module_artifact.exists() else specific_artifact

    if function_artifact.exists():
        return function_artifact
    if module_artifact.exists():
        return module_artifact

    # Default to the unique, module-scoped path when no artifact exists yet.
    return specific_artifact
