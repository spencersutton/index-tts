from typing import Self

from tensorboardX import SummaryWriter as _RuntimeSummaryWriter

from .utils import experimental

"""Contains a logger to push training logs to the Hub, using Tensorboard."""
is_summary_writer_available = ...

class HFSummaryWriter(_RuntimeSummaryWriter):
    @experimental
    def __new__(cls, *args, **kwargs) -> Self: ...
    def __init__(
        self,
        repo_id: str,
        *,
        logdir: str | None = ...,
        commit_every: float = ...,
        squash_history: bool = ...,
        repo_type: str | None = ...,
        repo_revision: str | None = ...,
        repo_private: bool | None = ...,
        path_in_repo: str | None = ...,
        repo_allow_patterns: list[str] | str | None = ...,
        repo_ignore_patterns: list[str] | str | None = ...,
        token: str | None = ...,
        **kwargs,
    ) -> None: ...
    def __exit__(self, exc_type, exc_val, exc_tb):  # -> None:

        ...
