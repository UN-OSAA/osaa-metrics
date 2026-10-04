"""``discover_indicators`` at the MCP tool boundary.

Pins the widget envelope {candidates, degraded, note}, the encoder floor (any
encoder failure, building or encoding, runs the keyword fallback flagged
"ranking unavailable", never a raw error), and the encoder gate (keyword-only
browse must not touch the encoder)."""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from osaa_metrics.config import load_settings
from osaa_metrics.mcp.providers import DataSourceProvider, EncoderProvider
from osaa_metrics.mcp.server import OSAAMetricsServer
from tests.conftest import DATA_AVAILABLE, NO_DATA_REASON

pytestmark = pytest.mark.skipif(not DATA_AVAILABLE, reason=NO_DATA_REASON)


def _server(encoder_factory) -> OSAAMetricsServer:
    return OSAAMetricsServer(
        data_source=DataSourceProvider(load_settings()),
        encoder_provider=EncoderProvider(factory=encoder_factory),
        models={},
    )


def _payload(result) -> dict:
    # House convention for tool results: the payload is the structured content.
    return result.structured_content


class _BoomEncoder:
    def encode(self, sentences, *, normalize_embeddings: bool = True):
        raise RuntimeError("weights half-downloaded")


@pytest.mark.asyncio
async def test_keyword_browse_never_constructs_encoder():
    def forbidden_factory():
        raise AssertionError("encoder must not be constructed for keyword-only")

    async with Client(_server(forbidden_factory)) as client:
        result = await client.call_tool("discover_indicators", {"keyword_query": "gdp"})
    data = _payload(result)
    assert set(data) == {"candidates", "degraded", "note"}
    assert data["degraded"] is False
    assert data["note"] is None


@pytest.mark.asyncio
async def test_broken_encoder_build_degrades_with_ranking_note():
    def broken_factory():
        raise RuntimeError("torch is broken on this machine")

    async with Client(_server(broken_factory)) as client:
        result = await client.call_tool(
            "discover_indicators", {"semantic_query": "school completion"}
        )
    data = _payload(result)
    assert data["note"] == "ranking unavailable"
    assert data["degraded"] is True
    assert isinstance(data["candidates"], list)


@pytest.mark.asyncio
async def test_encode_time_failure_degrades_with_ranking_note():
    async with Client(_server(lambda: _BoomEncoder())) as client:
        result = await client.call_tool(
            "discover_indicators", {"semantic_query": "school completion"}
        )
    data = _payload(result)
    assert data["note"] == "ranking unavailable"
    assert data["degraded"] is True


@pytest.mark.asyncio
async def test_rejected_query_is_not_persisted():
    async with Client(_server(lambda: _BoomEncoder())) as client:
        with pytest.raises(ToolError, match="contains control characters"):
            await client.call_tool(
                "discover_indicators", {"keyword_query": "bad\x00query"}
            )
        echo = _payload(await client.call_tool("open_discovery", {}))
    assert echo["keyword_query"] == ""  # the rejected string never landed
