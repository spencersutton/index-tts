import time
from pathlib import Path

from pyinstrument import Profiler
from pyinstrument.frame import Frame
from pyinstrument.frame_ops import delete_frame_from_tree
from pyinstrument.processors import ProcessorOptions, remove_unnecessary_self_time_nodes
from pyinstrument.renderers import HTMLRenderer


def remove_wrapper(frame: Frame, options: ProcessorOptions) -> Frame:
    # NOTE: pyinstrument's tree is mutated when deleting nodes.
    # Iterating directly over `frame.children` while mutating it can skip nodes.
    # Use an index-based loop so newly-spliced children are also visited.
    i = 0
    while i < len(frame.children):
        child = frame.children[i]
        if child.identifier in ["_call_impl", "compile_wrapper"]:
            delete_frame_from_tree(child, "children")
            # Don't increment i: there may now be new frames at this index.
            continue
        if child.identifier in ["_fn", "decorate_context"]:
            delete_frame_from_tree(child, "self_time")
            continue
        remove_wrapper(child, options)
        i += 1
    return frame


def generate_profile_report(profiler: Profiler) -> None:
    renderer = HTMLRenderer()
    renderer.preprocessors = [remove_unnecessary_self_time_nodes, remove_wrapper]
    Path(f"outputs/profile_{int(time.time())}.html").write_text(profiler.output(renderer))
