"""The two public entry points load the encoder once, before serving, and
never let a load failure stop the server from starting."""

from __future__ import annotations

import logging
import os
import types

import pytest

from osaa_metrics import mcp as mcp_pkg
from osaa_metrics.mcp.providers import EncoderProvider


@pytest.fixture(autouse=True)
def _hub_offline_flag_is_restored():
    """Both entry points write HF_HUB_OFFLINE straight into the real
    os.environ with a plain ``setdefault`` call, not through monkeypatch's
    tracked API, so monkeypatch's own restore stack cannot see that write
    and never reverses it — a value one test's entry-point call sets stays
    set for every test that runs after it in this process. This fixture
    reads the variable itself before each test and restores that exact
    state afterwards (absent stays absent), so no other test runs offline
    by accident."""
    previous = os.environ.get("HF_HUB_OFFLINE")
    yield
    if previous is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = previous


class _StubEncoder:
    def encode(self, sentences, *, normalize_embeddings: bool = True):
        return [[0.0] * 1024 for _ in sentences]


def _fake_server(factory, events):
    server = types.SimpleNamespace()
    server.encoder_provider = EncoderProvider(factory=factory)
    server.data_source = types.SimpleNamespace(
        settings=types.SimpleNamespace(bind_host="127.0.0.1", port=10000)
    )

    def run(**kwargs):
        events.append(("run", kwargs))

    server.run = run
    return server


def test_main_loads_the_encoder_before_serving(monkeypatch, capsys):
    events = []

    def factory():
        events.append(("load", {}))
        return _StubEncoder()

    monkeypatch.setattr(mcp_pkg, "build_server", lambda: _fake_server(factory, events))
    mcp_pkg.main()
    assert [name for name, _ in events] == ["load", "run"]
    captured = capsys.readouterr()
    assert "loading the encoder" in captured.err
    assert "encoder loaded in" in captured.err
    assert captured.out == ""


def test_main_serves_even_when_the_encoder_fails_to_load(monkeypatch, caplog, capsys):
    events = []

    def factory():
        raise RuntimeError("pinned files are not cached")

    monkeypatch.setattr(mcp_pkg, "build_server", lambda: _fake_server(factory, events))
    with caplog.at_level(logging.WARNING, logger="osaa_metrics.mcp"):
        mcp_pkg.main()
    assert [name for name, _ in events] == ["run"]
    assert "encoder" in caplog.text
    assert "pinned files are not cached" in caplog.text
    assert "encoder loaded in" not in capsys.readouterr().err


def test_boot_line_names_no_particular_client(capsys):
    """_load_encoder runs from both entry points, so its line is read by
    whichever client started the server. It must not name one."""
    mcp_pkg._load_encoder(_fake_server(_StubEncoder, []))
    err = capsys.readouterr().err
    assert "loading the encoder" in err
    assert "Claude Desktop" not in err


def test_run_http_loads_then_serves_with_the_bind_settings(monkeypatch):
    events = []

    def factory():
        events.append(("load", {}))
        return _StubEncoder()

    monkeypatch.setattr(
        mcp_pkg,
        "build_server",
        lambda encoder_factory=None: _fake_server(factory, events),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    mcp_pkg.run_http()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert events[0] == ("load", {})
    name, kwargs = events[1]
    assert name == "run"
    assert kwargs == {
        "transport": "streamable-http",
        "host": "127.0.0.1",
        "port": 10000,
    }


def test_run_http_forwards_an_injected_factory(monkeypatch):
    seen = {}

    def fake_build_server(encoder_factory=None):
        seen["factory"] = encoder_factory
        return _fake_server(encoder_factory, [])

    monkeypatch.setattr(mcp_pkg, "build_server", fake_build_server)
    sentinel_factory = _StubEncoder
    mcp_pkg.run_http(encoder_factory=sentinel_factory)
    assert seen["factory"] is sentinel_factory


def test_entry_points_forbid_hub_network_in_their_own_process(monkeypatch):
    """The hub library reads HF_HUB_OFFLINE at import time; the entry points
    set it before anything imports the library, and only in the server
    process — a notebook importing the package is untouched."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(mcp_pkg, "build_server", lambda: _fake_server(_StubEncoder, []))
    mcp_pkg.main()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
