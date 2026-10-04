"""FastMCP server entry point.

``build_server`` imports what it constructs inside the function body, so
importing ``osaa_metrics.mcp`` costs nothing beyond the package itself:
FastMCP, BSL, the providers and the theme load only when a server is actually
built.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osaa_metrics.mcp.server import OSAAMetricsServer

logger = logging.getLogger(__name__)


def build_server(encoder_factory=None) -> OSAAMetricsServer:
    """Construct the composing FastMCP server (shared by stdio + http).

    ``encoder_factory`` is the encoder-construction seam: the default builds
    the local encoder; a deployment that runs another backend passes a
    zero-argument factory here.

    No I/O happens here: the data connection is built lazily by
    ``DataSourceProvider`` on first use, so a bad source cannot kill the
    process before the MCP handshake (graceful boot); the encoder is loaded
    by the entry points after construction, never here."""
    from osaa_metrics.config import load_settings
    from osaa_metrics.encoders import LocalEncoder
    from osaa_metrics.mcp.providers import DataSourceProvider, EncoderProvider
    from osaa_metrics.mcp.server import OSAAMetricsServer
    from osaa_metrics.theme import register_theme

    register_theme()

    settings = load_settings()
    if encoder_factory is None:
        encoder_factory = LocalEncoder
    return OSAAMetricsServer(
        data_source=DataSourceProvider(settings),
        encoder_provider=EncoderProvider(factory=encoder_factory),
        models={},
    )


def _forbid_hub_network() -> None:
    """Tell the hub library to make no HTTP call in this process, unless the
    environment already set the variable itself. It reads the variable when
    it is imported, so this runs before the encoder load
    that imports it. Set here, in the server entry points, and not in the
    library: a notebook that imports the package keeps its own network
    policy."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _load_encoder(server) -> None:
    """Load the encoder once, before the server serves, so no chat message
    pays for it. A failure is logged and left to the discovery floor: every
    semantic search then runs as a keyword search flagged "ranking
    unavailable" until a later load succeeds. Boot never dies on it. Two
    lines bracket the load on stderr — one before, one after a successful
    load with the elapsed seconds — so the host's per-server log shows
    whether the load is still running."""
    print(  # noqa: T201  # stderr only; stdout carries the stdio transport's protocol frames
        "osaa-metrics: loading the encoder; the server answers its client after this finishes",
        file=sys.stderr,
        flush=True,
    )
    start = time.perf_counter()
    try:
        server.encoder_provider.get()
    except Exception:  # any load failure is the discovery floor's to report; the server must still start
        logger.warning(
            "encoder failed to load at start; semantic searches run as keyword searches",
            exc_info=True,
        )
    else:
        elapsed = time.perf_counter() - start
        print(  # noqa: T201  # stderr only; stdout carries the stdio transport's protocol frames
            f"osaa-metrics: encoder loaded in {elapsed:.1f}s; starting the server",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    """Launch the composing FastMCP server over stdio."""
    _forbid_hub_network()
    server = build_server()
    _load_encoder(server)
    server.run()


def run_http(encoder_factory=None) -> None:
    """Launch over streamable HTTP. Binds this-machine-only
    by default; deployments set OSAA_BIND_HOST=0.0.0.0 explicitly.
    ``encoder_factory`` is forwarded to ``build_server`` so a deployment can
    run another encoder through the same entry point."""
    _forbid_hub_network()
    server = build_server(encoder_factory)
    _load_encoder(server)
    settings = server.data_source.settings
    server.run(
        transport="streamable-http",
        host=settings.bind_host,
        port=settings.port,
    )
