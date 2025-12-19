from pathlib import Path
from typing import Any

from gradio.cli.commands.display import LivePanelDisplay

def run_command(
    live: LivePanelDisplay,
    name: str,
    pyproject_toml: dict[str, Any],
    suppress_demo_check: bool,
    generate_space: bool,
    generate_readme: bool,
    type_mode: str,
    _demo_path: Path,
    _demo_dir: Path,
    _readme_path: Path,
    space_url: str | None,
    _component_dir: Path,
    simple: bool = ...,
):  # -> None:
    ...
