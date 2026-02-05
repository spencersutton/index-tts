from pathlib import Path

from pyinstrument import Profiler
from pyinstrument.frame import Frame
from pyinstrument.frame_ops import delete_frame_from_tree
from pyinstrument.processors import ProcessorOptions
from pyinstrument.renderers import HTMLRenderer


def _next_profile_slot(out_dir: Path, slots: int = 5) -> int:
    """Infer the next rotation slot from existing output files.

    Picks the slot after the most recently modified `profile_{i}.html`.
    If none exist, returns 0.
    """

    newest: tuple[float, int] | None = None
    for i in range(slots):
        p = out_dir / f"profile_{i}.html"
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        # Break ties deterministically by preferring the higher slot.
        candidate = (mtime, i)
        if newest is None or candidate > newest:
            newest = candidate
    if newest is None:
        return 0
    return (newest[1] + 1) % slots


def remove_wrapper(frame: Frame, options: ProcessorOptions) -> Frame:
    i = 0
    squash = ["__get__", "generator_context", "decorate_context", "_wrapped_call_impl", "_call_impl", "wrapped_func"]
    to_self = ["_fn"]

    while i < len(frame.children):
        child = frame.children[i]
        if any(x in child.identifier or x in child.function for x in squash):
            delete_frame_from_tree(child, "children")
            continue

        if any(x in child.identifier or x in child.function for x in to_self):
            delete_frame_from_tree(child, "self_time")
            continue
        remove_wrapper(child, options)
        i += 1
    return frame


def generate_profile_report(profiler: Profiler) -> None:
    renderer = HTMLRenderer()
    renderer.preprocessors = [remove_wrapper]

    # Rotate through a fixed set of output files across program invocations.
    # This keeps the last N profiles without endlessly accumulating files.
    out_dir = Path("outputs")
    slot = _next_profile_slot(out_dir, slots=5)

    html = profiler.output(renderer)
    (out_dir / f"profile_{slot}.html").write_text(html)
