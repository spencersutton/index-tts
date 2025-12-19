from pathlib import Path

"""

Contains the functions that run when `gradio` is called from the command line. Specifically, allows

$ gradio app.py, to run app.py in reload mode where any changes in the app.py file or Gradio library reloads the demo.
$ gradio app.py my_demo, to use variable names other than "demo"
"""
reload_thread = ...

def main(
    demo_path: Path,
    demo_name: str = ...,
    watch_dirs: list[str] | None = ...,
    encoding: str = ...,
    watch_library: bool = ...,
):  # -> None:
    ...

if __name__ == "__main__": ...
