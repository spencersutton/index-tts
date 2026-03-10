# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast, type_check_only

from torch import Tensor
from torch.utils import cpp_extension

logger = logging.getLogger(__name__)

"""
Setting this param to a list has a problem of generating different compilation commands
(with diferent order of architectures) and leading to recompilation of fused kernels. 
Set it to empty stringo avoid recompilation and assign arch flags explicity in extra_cuda_cflags below
"""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

if TYPE_CHECKING:

    @type_check_only
    class CudaActivationModule(ABC):
        @abstractmethod
        def forward(self, inputs: Tensor, up_ftr: Tensor, down_ftr: Tensor, alpha: Tensor, beta: Tensor) -> Tensor: ...


def load() -> CudaActivationModule:
    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag: list[str] = []
    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home is None:
        raise RuntimeError("CUDA_HOME is not set; cannot build anti_alias_activation CUDA extension")

    _, bare_metal_major, _ = _get_cuda_bare_metal_version(cuda_home)
    if int(bare_metal_major) >= 11:
        cc_flag.extend(("-gencode", "arch=compute_80,code=sm_80"))

    # Build path
    srcpath = Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(
        name: str, sources: Sequence[str | Path], extra_cuda_flags: Sequence[str]
    ) -> CudaActivationModule:
        return cast(
            CudaActivationModule,
            cpp_extension.load(
                name=name,
                sources=list(sources),
                build_directory=buildpath,
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                    "--use_fast_math",
                    *extra_cuda_flags,
                    *cc_flag,
                ],
                verbose=True,
            ),
        )

    extra_cuda_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    sources = [srcpath / "anti_alias_activation.cpp", srcpath / "anti_alias_activation_cuda.cu"]
    return _cpp_extention_load_helper("anti_alias_activation_cuda", sources, extra_cuda_flags)


def _get_cuda_bare_metal_version(cuda_dir: str) -> tuple[str, str, str]:
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath: Path) -> None:
    try:
        buildpath.mkdir()
    except OSError:
        if not buildpath.is_dir():
            logger.error("Creation of the build directory %s failed", buildpath)
