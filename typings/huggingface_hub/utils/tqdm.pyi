import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tqdm.auto import tqdm as old_tqdm

"""Utility helpers to handle progress bars in `huggingface_hub`.

Example:
    1. Use `huggingface_hub.utils.tqdm` as you would use `tqdm.tqdm` or `tqdm.auto.tqdm`.
    2. To disable progress bars, either use `disable_progress_bars()` helper or set the
       environment variable `HF_HUB_DISABLE_PROGRESS_BARS` to 1.
    3. To re-enable progress bars, use `enable_progress_bars()`.
    4. To check whether progress bars are disabled, use `are_progress_bars_disabled()`.

NOTE: Environment variable `HF_HUB_DISABLE_PROGRESS_BARS` has the priority.

Example:
    ```py
    >>> from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars, tqdm

    # Disable progress bars globally
    >>> disable_progress_bars()

    # Use as normal `tqdm`
    >>> for _ in tqdm(range(5)):
    ...    pass

    # Still not showing progress bars, as `disable=False` is overwritten to `True`.
    >>> for _ in tqdm(range(5), disable=False):
    ...    pass

    >>> are_progress_bars_disabled()
    True

    # Re-enable progress bars globally
    >>> enable_progress_bars()

    # Progress bar will be shown !
    >>> for _ in tqdm(range(5)):
    ...   pass
    100%|███████████████████████████████████████| 5/5 [00:00<00:00, 117817.53it/s]
    ```

Group-based control:
    ```python
    # Disable progress bars for a specific group
    >>> disable_progress_bars("peft.foo")

    # Check state of different groups
    >>> assert not are_progress_bars_disabled("peft"))
    >>> assert not are_progress_bars_disabled("peft.something")
    >>> assert are_progress_bars_disabled("peft.foo"))
    >>> assert are_progress_bars_disabled("peft.foo.bar"))

    # Enable progress bars for a subgroup
    >>> enable_progress_bars("peft.foo.bar")

    # Check if enabling a subgroup affects the parent group
    >>> assert are_progress_bars_disabled("peft.foo"))
    >>> assert not are_progress_bars_disabled("peft.foo.bar"))

    # No progress bar for `name="peft.foo"`
    >>> for _ in tqdm(range(5), name="peft.foo"):
    ...     pass

    # Progress bar will be shown for `name="peft.foo.bar"`
    >>> for _ in tqdm(range(5), name="peft.foo.bar"):
    ...     pass
    100%|███████████████████████████████████████| 5/5 [00:00<00:00, 117817.53it/s]

    ```
"""
progress_bar_states: dict[str, bool] = ...

def disable_progress_bars(name: str | None = ...) -> None: ...
def enable_progress_bars(name: str | None = ...) -> None: ...
def are_progress_bars_disabled(name: str | None = ...) -> bool: ...
def is_tqdm_disabled(log_level: int) -> bool | None: ...

class tqdm(old_tqdm):
    def __init__(self, *args, **kwargs) -> None: ...
    def __delattr__(self, attr: str) -> None: ...

@contextmanager
def tqdm_stream_file(path: Path | str) -> Iterator[io.BufferedReader]: ...
