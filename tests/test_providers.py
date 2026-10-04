"""Direct unit tests for EncoderProvider + DataSourceProvider. The lazy-load
and caching invariants are pinned here, so a broken one surfaces at the unit
level instead of only through an MCP integration test.
"""

from __future__ import annotations

import polars as pl

from osaa_metrics.config import load_settings
from osaa_metrics.mcp.providers import (
    DataSourceProvider,
    EncoderProvider,
)

# --- EncoderProvider ---


class _FakeEncoder:
    """Stand-in for the BGE-M3 encoder — keeps the real model out of the test."""

    def encode(self, sentences, *, normalize_embeddings: bool = True):
        return [[0.0] * 1024]


def test_encoder_provider_lazy_loads_only_on_get():
    """Construction must not trigger the factory; the factory only runs on
    the first ``.get()`` call, so building a server never pays for an encoder
    nobody asked for."""
    calls = []

    def fake_factory():
        calls.append(1)
        return _FakeEncoder()

    p = EncoderProvider(factory=fake_factory)
    assert calls == []  # construction is cheap
    enc = p.get()  # first call runs factory
    assert calls == [1]
    assert isinstance(enc, _FakeEncoder)


def test_encoder_provider_caches_after_first_get():
    """Subsequent ``.get()`` calls must return the same instance — not
    re-invoke the factory, which would reload BGE-M3 on every semantic
    search."""
    calls = []

    def fake_factory():
        calls.append(1)
        return _FakeEncoder()

    p = EncoderProvider(factory=fake_factory)
    enc1 = p.get()
    enc2 = p.get()
    enc3 = p.get()
    assert calls == [1]  # factory ran exactly once
    assert enc1 is enc2 is enc3  # same instance every call


def test_encoder_provider_refuses_a_factory_that_returns_none():
    """A factory that returns ``None`` must not be treated as a built
    encoder — the caller (discovery) would otherwise take ``None`` for a
    real encoder and run a silent keyword fallback that nothing flags."""
    p = EncoderProvider(factory=lambda: None)
    try:
        p.get()
        raised = False
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised
    assert "returned None" in msg

    # Nothing cached: the next get() raises again instead of returning None.
    try:
        p.get()
        raised_again = False
    except RuntimeError:
        raised_again = True
    assert raised_again


def test_encoder_provider_reset_clears_cache():
    """``reset()`` is the test-seam — after it the next ``.get()`` must
    re-invoke the factory."""
    calls = []

    def fake_factory():
        calls.append(1)
        return _FakeEncoder()

    p = EncoderProvider(factory=fake_factory)
    p.get()
    p.reset()
    p.get()
    assert calls == [1, 1]  # factory ran twice (once before reset, once after)


# --- DataSourceProvider catalogue cache ---


def test_data_source_provider_catalogue_lazy_and_cached(monkeypatch):
    """``get_catalogue()`` must lazy-load on first call and return the cached
    frame on subsequent calls."""
    calls = []

    def fake_load_embedding_cache(*, con):
        calls.append(con)
        return pl.DataFrame({"indicator_code": ["X.A", "X.B"]})

    monkeypatch.setattr(
        "osaa_metrics.mcp.providers.data_source.load_embedding_cache",
        fake_load_embedding_cache,
    )

    p = DataSourceProvider(load_settings(env={}), con_factory=lambda s: "con-A")
    assert p._catalogue_df is None

    df1 = p.get_catalogue()
    df2 = p.get_catalogue()

    assert len(calls) == 1, "load_embedding_cache must run exactly once"
    assert df1 is df2
    assert df1["indicator_code"].to_list() == ["X.A", "X.B"]


def test_data_source_provider_reset_clears_catalogue(monkeypatch):
    """``reset()`` must drop the cached catalogue — used by tests to start clean."""
    monkeypatch.setattr(
        "osaa_metrics.mcp.providers.data_source.load_embedding_cache",
        lambda *, con: pl.DataFrame({"indicator_code": ["X.A"]}),
    )
    p = DataSourceProvider(load_settings(env={}), con_factory=lambda s: "con-A")
    p.get_catalogue()
    assert p._catalogue_df is not None
    p.reset()
    assert p._catalogue_df is None


# --- DataSourceProvider lazy connection (built from Settings) ---


def test_get_con_lazy_and_cached():
    calls = []

    def fake_factory(settings):
        calls.append(settings)
        return object()

    p = DataSourceProvider(load_settings(env={}), con_factory=fake_factory)
    assert calls == []  # nothing at construction (graceful boot)
    con1 = p.get_con()
    con2 = p.get_con()
    assert con1 is con2
    assert len(calls) == 1  # factory ran exactly once


def test_get_con_failure_is_named_tool_error(tmp_path):
    from osaa_metrics.mcp.tools import ToolValidationError

    settings = load_settings(
        env={
            "OSAA_DATA_MASTER_URL": str(tmp_path / "missing.parquet"),
            "OSAA_DATA_META_URL": str(tmp_path / "missing.parquet"),
        }
    )

    def broken_factory(settings):
        raise FileNotFoundError("no such parquet")

    p = DataSourceProvider(settings, con_factory=broken_factory)
    try:
        p.get_con()
        raised = False
    except ToolValidationError as e:
        raised = True
        msg = str(e)
    assert raised
    assert "OSAA_DATA_MASTER_URL" in msg  # names the knob to fix
    assert "missing.parquet" in msg  # names the value that failed


def test_reset_drops_connection_and_catalogue():
    p = DataSourceProvider(load_settings(env={}), con_factory=lambda s: object())
    first = p.get_con()
    p.reset()
    second = p.get_con()
    assert first is not second


# --- DI-shape smoke check ---


def test_providers_module_exports_only_classes():
    """The module surface is the provider classes with no pre-instantiated
    instances: each consumer (OSAAMetricsServer, the test suite) constructs
    its own. A module-level instance would share one connection and one
    encoder across every server built in the process."""
    from osaa_metrics.mcp import providers

    assert hasattr(providers, "DataSourceProvider")
    assert hasattr(providers, "EncoderProvider")
    assert not hasattr(providers, "data_source_provider")
    assert not hasattr(providers, "encoder_provider")


# --- build_server seam: default encoder and injected factory ---


def test_build_server_defaults_to_the_local_encoder(monkeypatch, tmp_path):
    """With no factory injected, the public server's encoder factory is the
    local encoder class — there is no configuration that selects another.
    Checked on the factory, not by building it: construction loads the real
    weights."""
    monkeypatch.setenv("OSAA_DATA_MASTER_URL", str(tmp_path / "absent.parquet"))
    monkeypatch.setenv("OSAA_DATA_META_URL", str(tmp_path / "absent.parquet"))
    from osaa_metrics.encoders.local import LocalEncoder
    from osaa_metrics.mcp import build_server

    server = build_server()
    assert server.encoder_provider._factory is LocalEncoder


def test_build_server_accepts_encoder_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("OSAA_DATA_MASTER_URL", str(tmp_path / "absent.parquet"))
    monkeypatch.setenv("OSAA_DATA_META_URL", str(tmp_path / "absent.parquet"))
    from osaa_metrics.mcp import build_server

    sentinel = object()
    server = build_server(encoder_factory=lambda: sentinel)
    assert server.encoder_provider.get() is sentinel
