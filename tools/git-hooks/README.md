# Git Hooks

This directory contains git hooks for the project to automate common tasks.

## Available Hooks

### pre-commit
Automatically runs `uv run ruff format` before each commit to ensure code formatting consistency.

### post-checkout
Automatically runs `uv sync` after checking out branches to keep dependencies in sync.

## Installation

To install the git hooks, run:

```bash
bash tools/git-hooks/install.sh
```

Or from this directory:

```bash
bash install.sh
```

## Uninstallation

To remove the hooks, simply delete them from `.git/hooks/`:

```bash
rm .git/hooks/pre-commit .git/hooks/post-checkout
```

## Manual Usage

You can also run the hooks manually:

```bash
# Format code
bash tools/git-hooks/pre-commit

# Sync dependencies
bash tools/git-hooks/post-checkout 0 0 1
```
