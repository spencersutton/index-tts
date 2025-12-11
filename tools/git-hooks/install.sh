#!/bin/bash
# Git hooks installer script
# This script installs the git hooks into .git/hooks/

set -e

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$REPO_ROOT" ]; then
	echo "❌ Error: Not in a git repository"
	exit 1
fi

HOOKS_DIR="$REPO_ROOT/.git/hooks"
SOURCE_DIR="$REPO_ROOT/tools/git-hooks"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
	echo "❌ Error: Source directory $SOURCE_DIR not found"
	exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

echo "Installing git hooks..."

# Install pre-commit hook
if [ -f "$SOURCE_DIR/pre-commit" ]; then
	cp "$SOURCE_DIR/pre-commit" "$HOOKS_DIR/pre-commit"
	chmod +x "$HOOKS_DIR/pre-commit"
	echo "✅ Installed pre-commit hook"
else
	echo "⚠️  pre-commit hook not found"
fi

# Install post-checkout hook
if [ -f "$SOURCE_DIR/post-checkout" ]; then
	cp "$SOURCE_DIR/post-checkout" "$HOOKS_DIR/post-checkout"
	chmod +x "$HOOKS_DIR/post-checkout"
	echo "✅ Installed post-checkout hook"
else
	echo "⚠️  post-checkout hook not found"
fi

echo ""
echo "🎉 Git hooks installed successfully!"
echo ""
echo "Installed hooks:"
echo "  • pre-commit: Runs 'uv run ruff format' before each commit"
echo "  • post-checkout: Runs 'uv sync' after branch checkout"
echo ""
echo "To uninstall, simply delete the files in $HOOKS_DIR"
