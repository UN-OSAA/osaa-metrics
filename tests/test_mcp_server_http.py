"""Tests for the build_server() factory and the server it builds."""

from __future__ import annotations

import pytest


@pytest.mark.needs_data
def test_build_server_returns_composing_server() -> None:
    from osaa_metrics.mcp import build_server
    from osaa_metrics.mcp.server import OSAAMetricsServer

    assert isinstance(build_server(), OSAAMetricsServer)


@pytest.mark.needs_data
def test_health_route_registered() -> None:
    from osaa_metrics.mcp import build_server

    app = build_server().http_app()
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/health" in paths


def test_build_server_boots_without_touching_data_source(monkeypatch, tmp_path):
    """A bad data source must not kill boot — the error is deferred to first
    tool call."""
    monkeypatch.setenv("OSAA_DATA_MASTER_URL", str(tmp_path / "absent.parquet"))
    monkeypatch.setenv("OSAA_DATA_META_URL", str(tmp_path / "absent.parquet"))
    from osaa_metrics.mcp import build_server

    server = build_server()  # must NOT raise
    assert server.name == "osaa-metrics"


def test_con_property_raises_named_error_on_bad_source(monkeypatch, tmp_path):
    monkeypatch.setenv("OSAA_DATA_MASTER_URL", str(tmp_path / "absent.parquet"))
    monkeypatch.setenv("OSAA_DATA_META_URL", str(tmp_path / "absent.parquet"))
    from osaa_metrics.mcp import build_server
    from osaa_metrics.mcp.tools import ToolValidationError

    server = build_server()
    try:
        _ = server.con
        raised = False
    except ToolValidationError as e:
        raised = True
        assert "OSAA_DATA_MASTER_URL" in str(e)
    assert raised
