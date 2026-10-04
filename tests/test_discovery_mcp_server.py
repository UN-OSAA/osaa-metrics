"""DiscoveryMCPServer — OOP MCP shape, constructor DI, no BSL imports."""

from __future__ import annotations

import subprocess

import pytest
from fastmcp import FastMCP

from osaa_metrics.mcp_discovery.server import DiscoveryMCPServer


def test_discovery_mcp_server_is_fastmcp_subclass():
    assert issubclass(DiscoveryMCPServer, FastMCP)


def test_discovery_server_name_is_osaa_discovery(
    mock_data_source_provider, mock_encoder_provider
):
    srv = DiscoveryMCPServer(
        data_source=mock_data_source_provider,
        encoder_provider=mock_encoder_provider,
        models={},
    )
    assert srv.name == "osaa-discovery"


def test_discovery_subpackage_has_no_bsl_imports():
    """Discovery stays BSL-free: nothing under ``mcp_discovery`` imports
    ``boring_semantic_layer``. Matches real import statements only (anchored at
    line start, so docstring references are skipped)."""
    result = subprocess.run(
        [
            "grep",
            "-rln",
            "-E",
            r"^(from|import) boring_semantic_layer",
            "src/osaa_metrics/mcp_discovery/",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "", f"BSL imports found:\n{result.stdout}"


def test_discovery_server_stores_constructor_deps(
    mock_con, mock_data_source_provider, mock_encoder_provider
):
    srv = DiscoveryMCPServer(
        data_source=mock_data_source_provider,
        encoder_provider=mock_encoder_provider,
        models={},
    )
    assert srv.con is mock_con
    assert srv.encoder_provider is mock_encoder_provider


@pytest.mark.asyncio
async def test_discovery_server_registers_three_tools(
    mock_data_source_provider, mock_encoder_provider
):
    srv = DiscoveryMCPServer(
        data_source=mock_data_source_provider,
        encoder_provider=mock_encoder_provider,
        models={},
    )
    tool_names = {t.name for t in await srv.list_tools()}
    assert "open_discovery" in tool_names
    assert "discover_indicators" in tool_names
    assert "save_core_schema" in tool_names


def test_discovery_server_does_not_call_encoder_on_construction(
    mock_data_source_provider,
):
    """Lazy encoder: constructing the server must NOT invoke the provider's
    factory. The factory only runs on the first semantic discover_indicators."""
    from osaa_metrics.mcp.providers import EncoderProvider

    calls = []

    def fake_factory():
        calls.append(1)

        class _E:
            def encode(self, sentences, *, normalize_embeddings=True):
                import numpy as np

                return np.zeros((len(sentences), 1024), dtype="float32")

        return _E()

    provider = EncoderProvider(factory=fake_factory)
    DiscoveryMCPServer(
        data_source=mock_data_source_provider, encoder_provider=provider, models={}
    )
    assert calls == []  # factory must not run at construction time
