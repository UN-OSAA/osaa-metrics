"""The MCP tool registry is a contract surface.

Every tool the server exposes is relied on by agents; a tool accidentally
added, renamed, or dropped should fail loudly here rather than surface as a
broken client. The expected names live in
``tests/fixtures/mcp_registry.json`` — when the registry legitimately
changes, update that file in the same commit as the change that moves it.

Construction mirrors the DI-shape tests: an in-memory connection and a stub
encoder, so the check needs no local data and no network — it holds on a
fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from osaa_metrics.mcp.server import OSAAMetricsServer

_MANIFEST = Path(__file__).parent / "fixtures" / "mcp_registry.json"


@pytest.mark.asyncio
async def test_tool_registry_matches_manifest(
    mock_data_source_provider, mock_encoder_provider
) -> None:
    server = OSAAMetricsServer(
        data_source=mock_data_source_provider,
        encoder_provider=mock_encoder_provider,
        models={},
    )
    expected = json.loads(_MANIFEST.read_text())
    async with Client(server) as client:
        tools = await client.list_tools()
    assert sorted(tool.name for tool in tools) == expected
