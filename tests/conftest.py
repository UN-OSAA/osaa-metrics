"""Shared test fixtures.

The two session-shaped fixtures, each modelled on its upstream test suite:

- ``ctx`` — a freshly-constructed FastMCP ``Context`` with a ``MagicMock`` session.
  Use this for unit-level tests of state mutators / readers that take a
  ``ctx: Context`` arg directly. Mirrors
  ``fastmcp/tests/server/test_context.py::TestContextState``.

- ``client`` — an in-memory ``Client(server)`` connected to the real
  osaa-metrics MCP server. Use this for tool-roundtrip tests. Mirrors
  ``boring_semantic_layer/tests/test_semantic_mcp.py``. Multiple
  ``client.call_tool(...)`` calls within one ``async with`` block share the
  same session, so state mutations are observable across calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest_asyncio
from fastmcp import Client, Context, FastMCP

# Route data-backed tests through the local ``data/`` parquet mirror so pytest
# doesn't hit the data host's rate limit. Devs override by exporting
# OSAA_DATA_MASTER_URL / OSAA_DATA_META_URL before invoking pytest.
_DATA_DIR = Path(__file__).parent.parent / "data"
_SOURCES_CHOSEN = "OSAA_DATA_MASTER_URL" in os.environ
_MIRROR_PRESENT = (_DATA_DIR / "master.parquet").is_file() and (
    _DATA_DIR / "meta.parquet"
).is_file()
DATA_AVAILABLE = _SOURCES_CHOSEN or _MIRROR_PRESENT
NO_DATA_REASON = (
    "no local parquet mirror; export OSAA_DATA_MASTER_URL/OSAA_DATA_META_URL "
    "to run against another source"
)

if not _SOURCES_CHOSEN and _MIRROR_PRESENT:
    os.environ["OSAA_DATA_MASTER_URL"] = str(_DATA_DIR / "master.parquet")
    os.environ["OSAA_DATA_META_URL"] = str(_DATA_DIR / "meta.parquet")

import pytest  # noqa: E402  # imports follow the environment setup every test module relies on

from osaa_metrics.mcp import session_state  # noqa: E402
from osaa_metrics.mcp.providers import EncoderProvider  # noqa: E402
from osaa_metrics.mcp.server import OSAAMetricsServer  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Turn ``needs_data`` into a skip when the local mirror is absent."""
    if DATA_AVAILABLE:
        return
    skip_no_data = pytest.mark.skip(reason=NO_DATA_REASON)
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip_no_data)


@pytest.fixture
def mock_con():
    """Lightweight in-process ibis backend for DI-shape tests that don't
    issue any queries against real data."""
    import ibis

    return ibis.duckdb.connect(":memory:")


@pytest.fixture
def mock_data_source_provider(mock_con):
    """DataSourceProvider wired to the in-process ``mock_con`` backend, for
    DI-shape tests that need a provider but must not touch real data."""
    from osaa_metrics.config import load_settings
    from osaa_metrics.mcp.providers import DataSourceProvider

    return DataSourceProvider(load_settings(), con_factory=lambda settings: mock_con)


@pytest.fixture
def mock_encoder_provider():
    """EncoderProvider wrapping a stub encoder — returns a fixed zero vector.
    Sidesteps BGE-M3 download for tests that only check constructor wiring."""
    import numpy as np

    class StubEncoder:
        def encode(self, sentences, *, normalize_embeddings: bool = True):
            return np.zeros((len(sentences), 1024), dtype="float32")

    return EncoderProvider(factory=lambda: StubEncoder())


@pytest_asyncio.fixture
async def ctx():
    """Fresh Context with an isolated session; drops the runtime cache and
    recipe entries that session created on teardown."""
    server = FastMCP("test")
    mock_session = MagicMock()
    async with Context(fastmcp=server, session=mock_session) as c:
        try:
            yield c
        finally:
            # Drop any cache/recipe entry this ctx created so other tests start clean.
            session_state._cache_by_session.pop(c.session_id, None)
            session_state._recipe_by_session.pop(c.session_id, None)


@pytest_asyncio.fixture
async def client():
    """In-memory FastMCP Client wired to a fresh OSAAMetricsServer.

    Each call to this fixture builds a fresh server; multiple tool calls within
    the yielded client share one session_id. The connection is opened via the
    settings-driven DataSourceProvider (local mirror when present, see
    OSAA_DATA_MASTER_URL/OSAA_DATA_META_URL above); the encoder is the local
    one, wired through EncoderProvider exactly as build_server wires it.
    """
    if not DATA_AVAILABLE:
        pytest.skip(NO_DATA_REASON)

    from osaa_metrics.config import load_settings
    from osaa_metrics.encoders.local import LocalEncoder
    from osaa_metrics.mcp.providers import DataSourceProvider

    settings = load_settings()
    server = OSAAMetricsServer(
        data_source=DataSourceProvider(settings),
        encoder_provider=EncoderProvider(factory=LocalEncoder),
        models={},
    )
    async with Client(server) as c:
        yield c
    # Defensive: clear our cache + recipe dicts so MagicMock session_ids from
    # this client don't leak into the next test's lookups.
    session_state._cache_by_session.clear()
    session_state._recipe_by_session.clear()
