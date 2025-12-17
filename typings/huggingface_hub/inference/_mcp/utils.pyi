from typing import TYPE_CHECKING

from mcp import types as mcp_types

"""
Utility functions for MCPClient and Tiny Agents.

Formatting utilities taken from the JS SDK: https://github.com/huggingface/huggingface.js/blob/main/packages/mcp-client/src/ResultFormatter.ts.
"""
if TYPE_CHECKING: ...

def format_result(result: mcp_types.CallToolResult) -> str: ...
