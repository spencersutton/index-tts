"""Deprecated benchmark RTF test.

The historical `tools.benchmark.BenchmarkResult` helper no longer exists in the
current codebase. This file is kept as a placeholder so older references don't
break test collection.
"""

import pytest

pytest.skip(
    "Deprecated: BenchmarkResult helper was removed from the codebase.",
    allow_module_level=True,
)
