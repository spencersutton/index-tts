from torchaudio._internal.module_utils import dropping_class_io_support, dropping_io_support

from . import (
    _extension,
    backend,
    compliance,
    datasets,
    functional,
    io,
    kaldi_io,
    models,
    pipelines,
    sox_effects,
    transforms,
    utils,
)
from ._backend import (
    AudioMetaData as _AudioMetaData,
)
from ._backend import (
    get_audio_backend as _get_audio_backend,
)
from ._backend import (
    info as _info,
)
from ._backend import (
    list_audio_backends as _list_audio_backends,
)
from ._backend import (
    load,
    save,
)
from ._backend import (
    set_audio_backend as _set_audio_backend,
)
from ._torchcodec import load_with_torchcodec, save_with_torchcodec
from .version import __version__, git_version

AudioMetaData = ...
get_audio_backend = ...
info = ...
list_audio_backends = ...
set_audio_backend = ...
__all__ = [
    "AudioMetaData",
    "compliance",
    "datasets",
    "functional",
    "get_audio_backend",
    "info",
    "io",
    "kaldi_io",
    "list_audio_backends",
    "load",
    "load_with_torchcodec",
    "models",
    "pipelines",
    "save",
    "save_with_torchcodec",
    "set_audio_backend",
    "sox_effects",
    "transforms",
    "utils",
]
