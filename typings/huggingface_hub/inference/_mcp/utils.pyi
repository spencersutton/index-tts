from mcp import types as mcp_types

"""
Utility functions for MCPClient and Tiny Agents.

Formatting utilities taken from the JS SDK: https://github.com/huggingface/huggingface.js/blob/main/packages/mcp-client/src/ResultFormatter.ts.
"""

def format_result(result: mcp_types.CallToolResult) -> str: ...
