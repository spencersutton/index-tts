import json
import os
import subprocess
import sys
from pathlib import Path

print_nodes = False
entry_point = "webui.py"

output = subprocess.run(["ruff", "analyze", "graph"], capture_output=True)
files = json.loads(output.stdout.decode())

visited = set()


def dfs(node: str, depth: int) -> None:
    if node in visited:
        return
    visited.add(node)
    if print_nodes:
        print("  " * depth, node, sep="")
    for dep in files[node]:
        dfs(dep, depth + 1)


dfs(entry_point, 0)


result = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard", "*.py"], cwd=".", capture_output=True, text=True
)

tracked_files = subprocess.run(["git", "ls-files", "*.py"], cwd=".", capture_output=True, text=True)

all_files = sorted(set(tracked_files.stdout.strip().split("\n") + result.stdout.strip().split("\n")))

python_files = set([
    f
    for f in all_files
    if f.endswith(".py")
    and f
    and not f.startswith("tests/")
    and not f.startswith("tools/")
    and f not in {"indextts/cli.py", "sitecustomize.py"}
    and not f.endswith("__init__.py")
])

unused = sorted(list(python_files - visited - {entry_point}))


def is_ignored(path_list: list[str]) -> set[str]:
    """
    Sends a list of paths to 'git check-ignore' and returns
    a set of paths that Git considers ignored.
    """
    if not path_list:
        return set()

    try:
        # --stdin allows us to check multiple files in one process
        # --no-index allows it to work even on files not yet tracked
        cmd = ["git", "check-ignore", "--stdin", "--no-index"]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, _ = process.communicate(input="\n".join(path_list))
        return set(stdout.splitlines())
    except FileNotFoundError:
        # Git isn't installed or we aren't in a repo
        return set()


def find_lone_init_dirs(root_path: str = ".") -> list[str]:
    matches = []

    for root, dirs, files in os.walk(root_path):
        # Exclude the .git directory itself
        if ".git" in dirs:
            dirs.remove(".git")
        if ".venv" in dirs:
            dirs.remove(".venv")

        # Create full paths for checking
        full_paths = [str(Path(root) / f) for f in files]

        # Ask Git which of these files are ignored
        ignored_files = is_ignored(full_paths)

        # Filter files to only those NOT ignored by Git
        visible_files = [f for f in full_paths if f not in ignored_files]

        # Apply logic: Exactly one visible file, and it must be __init__.py
        if len(visible_files) == 1 and Path(visible_files[0]).name == "__init__.py":
            # Ensure no subdirectories are visible (not ignored)
            full_dirs = [str(Path(root) / d) for d in dirs]
            ignored_dirs = is_ignored(full_dirs)
            visible_dirs = [d for d in full_dirs if d not in ignored_dirs]

            if len(visible_dirs) == 0:
                matches.append(root)

    return matches


if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
    results = find_lone_init_dirs()
    if results:
        print("\n".join(results))
else:
    for f in unused:
        print(f)
