"""Tests for session_state and the state-touching MCP tools.

Two patterns (see ``conftest.py``):
- ``ctx`` fixture for unit-level mutator/reader tests (Pattern 1 from
  FastMCP's own tests).
- ``client`` fixture for tool-roundtrip integration tests (Pattern 2 from
  BSL's MCP tests). ``open_discovery`` returns the recipe dict so any
  follow-up assertion just reads its ``structured_content``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, Context, FastMCP

from osaa_metrics.mcp import session_state

# =========================================================================
# Unit tests — session_state module via the ctx fixture
# =========================================================================


@pytest.mark.asyncio
async def test_get_state_returns_recipe_defaults(ctx) -> None:
    """get_state(ctx) returns the SessionRecipe dict shape with defaults
    when nothing has been written yet."""
    s = await session_state.get_state(ctx)
    assert s == {
        "semantic_query": "",
        "keyword_query": "",
        "committed_core_table": [],
        "last_saved_at": None,
        "core_model_yaml": None,
        "last_validation_errors": None,
        "last_analysis": None,
        "core_model_certified_map": None,
    }


@pytest.mark.asyncio
async def test_get_cache_returns_singleton_per_session(ctx) -> None:
    """get_cache(ctx) returns the same RuntimeCache on repeat calls for the
    same session. Cache fields default to None."""
    c1 = session_state.get_cache(ctx)
    c2 = session_state.get_cache(ctx)
    assert c1 is c2
    assert c1.core_tbl_handle is None
    assert c1.core_semantic_table is None


@pytest.mark.asyncio
async def test_set_queries_persists_strings(ctx) -> None:
    await session_state.set_queries(ctx, "trade balance", "LCU")
    s = await session_state.get_state(ctx)
    assert s["semantic_query"] == "trade balance"
    assert s["keyword_query"] == "LCU"


@pytest.mark.asyncio
async def test_set_queries_handles_none_as_empty(ctx) -> None:
    await session_state.set_queries(ctx, None, None)
    s = await session_state.get_state(ctx)
    assert s["semantic_query"] == ""
    assert s["keyword_query"] == ""


@pytest.mark.asyncio
async def test_cache_lru_cap_evicts_oldest_session() -> None:
    """The kitchen dict is bounded by ``_CACHE_MAX_SESSIONS``. Accessing a
    session marks it most-recently-used; once the cap is exceeded, the
    least-recently-used session is dropped. The recipe dict is NOT capped:
    it is the source of truth every kitchen rebuild reads."""
    from unittest.mock import MagicMock

    from fastmcp import Context, FastMCP

    server = FastMCP("test-lru")
    cap = session_state._CACHE_MAX_SESSIONS

    # Snapshot pre-test state so a leftover cache from another test doesn't
    # spuriously fail this one.
    session_state._cache_by_session.clear()
    session_state._recipe_by_session.clear()

    contexts: list[Context] = []
    try:
        # Open cap + 2 sessions; each get_cache writes one slot.
        for _ in range(cap + 2):
            session = MagicMock()
            ctx = await Context(fastmcp=server, session=session).__aenter__()
            contexts.append(ctx)
            session_state.get_cache(ctx)

        # Cap upheld.
        assert len(session_state._cache_by_session) == cap

        # The two oldest sessions are gone; the most recent ``cap`` survive.
        evicted_ids = {c.session_id for c in contexts[:2]}
        survivors = set(session_state._cache_by_session.keys())
        assert evicted_ids.isdisjoint(survivors)
        assert {c.session_id for c in contexts[2:]} == survivors
    finally:
        for c in contexts:
            await c.__aexit__(None, None, None)
        session_state._cache_by_session.clear()
        session_state._recipe_by_session.clear()


@pytest.mark.asyncio
async def test_save_core_schema_stores_copy_not_reference(ctx) -> None:
    rows = [{"indicator_code": "X", "var_name": "x"}]
    await session_state.save_core_schema(ctx, rows, "2026-05-04T10:00:00+00:00")
    rows[0]["var_name"] = "MUTATED"
    stored = (await session_state.get_state(ctx))["committed_core_table"]
    assert stored[0]["var_name"] == "x"  # copy, not the mutated reference


@pytest.mark.asyncio
async def test_invalidate_after_save_zeros_downstream(ctx) -> None:
    # Seed the recipe + cache to look post-load.
    await session_state.set_core_semantic_table(ctx, object(), "yaml-text")
    cache = session_state.get_cache(ctx)

    await session_state.invalidate_after_save(ctx, models={})

    s = await session_state.get_state(ctx)
    assert s["core_model_yaml"] is None
    assert s["last_analysis"] is None
    assert cache.core_semantic_table is None


@pytest.mark.asyncio
async def test_invalidate_after_model_load_preserves_core_tbl(ctx) -> None:
    cache = session_state.get_cache(ctx)
    cache.core_tbl_handle = object()
    await session_state.set_core_semantic_table(ctx, object(), "yaml-text")

    await session_state.invalidate_after_model_load(ctx)

    s = await session_state.get_state(ctx)
    assert cache.core_tbl_handle is not None
    assert s["core_model_yaml"] == "yaml-text"
    assert cache.core_semantic_table is not None


@pytest.mark.asyncio
async def test_session_isolation_via_distinct_sessions() -> None:
    """Two MagicMock sessions = two slots. Mirrors FastMCP's own
    ``test_context_state_session_isolation``."""
    server = FastMCP("test")
    alice_session = MagicMock()
    bob_session = MagicMock()

    async with Context(fastmcp=server, session=alice_session) as alice:
        await session_state.set_queries(alice, "alice-query", "")

    async with Context(fastmcp=server, session=bob_session) as bob:
        bob_state = await session_state.get_state(bob)
        assert bob_state["semantic_query"] == ""

    async with Context(fastmcp=server, session=alice_session) as alice2:
        alice_state = await session_state.get_state(alice2)
        assert alice_state["semantic_query"] == "alice-query"


@pytest.mark.asyncio
async def test_recipe_is_json_round_trippable(ctx) -> None:
    """The recipe survives json.dumps/loads after queries, a Save and a core
    semantic table: it goes back to the client as structured_content, so every
    field on it has to be JSON-safe."""
    await session_state.set_queries(ctx, "trade", "africa")
    await session_state.save_core_schema(
        ctx,
        [{"indicator_code": "NY.GDP.MKTP.CD", "var_name": "gdp"}],
        "2026-05-04T10:00:00+00:00",
    )
    await session_state.set_core_semantic_table(ctx, object(), "yaml-text")

    s = await session_state.get_state(ctx)
    round_tripped = json.loads(json.dumps(s))
    assert round_tripped == s


@pytest.mark.asyncio
async def test_reset_drops_cache_entry(ctx) -> None:
    """reset(ctx) clears the cache entry and the recipe — the next
    get_cache returns a fresh RuntimeCache with no core table handle, and
    the next get_state shows no committed core table schema."""
    cache = session_state.get_cache(ctx)
    cache.core_tbl_handle = object()

    await session_state.reset(ctx, models={})

    s = await session_state.get_state(ctx)
    assert s["semantic_query"] == ""
    assert s["committed_core_table"] == []
    fresh_cache = session_state.get_cache(ctx)
    assert fresh_cache.core_tbl_handle is None


# =========================================================================
# Integration tests — through the FastMCP Client
# =========================================================================


@pytest.mark.asyncio
async def test_composing_server_registers_expected_tools(client) -> None:
    """The composing OSAAMetricsServer surface = discovery sub-server tools +
    mounted BSL tools (list_models, get_model, etc.) + composing-layer tools."""
    tools = await client.list_tools()
    names = {t.name for t in tools}
    # Discovery sub-server
    assert "open_discovery" in names
    assert "discover_indicators" in names
    assert "save_core_schema" in names
    # Composing layer
    assert "build_core_table" in names
    assert "define_semantic_model" in names
    assert "query_and_chart" in names
    assert "build_core_summary" in names
    assert "download_session" in names
    assert "reset_state" in names
    assert "load_session" in names
    assert "get_bsl_schema" in names
    assert "get_session_schema" in names
    # Mounted BSL
    assert "list_models" in names
    assert "query_model" in names


@pytest.mark.asyncio
async def test_open_discovery_with_no_seed_returns_state(client) -> None:
    # Seed via the tool first so we observe persistence across two calls in
    # one client session.
    await client.call_tool(
        "open_discovery",
        {"initial_semantic_query": "foo", "initial_keyword_query": "bar"},
    )
    result = await client.call_tool("open_discovery", {})
    data = result.structured_content
    assert data["semantic_query"] == "foo"
    assert data["keyword_query"] == "bar"


@pytest.mark.asyncio
async def test_open_discovery_with_seed_overwrites_only_queries(client) -> None:
    """Seeding one initial_*_query overwrites that query string and leaves the
    other one as it stood — open_discovery seeds state, it never clears it."""
    # First call seeds initial state.
    await client.call_tool(
        "open_discovery",
        {"initial_semantic_query": "trade balance", "initial_keyword_query": "LCU"},
    )
    # Second call overrides only the semantic_query.
    result = await client.call_tool(
        "open_discovery", {"initial_semantic_query": "fiscal policy"}
    )
    data = result.structured_content
    assert data["semantic_query"] == "fiscal policy"
    assert data["keyword_query"] == "LCU"  # untouched


@pytest.mark.asyncio
async def test_reset_state_then_open_discovery_zeros_state(client) -> None:
    """Starting clean takes an explicit reset_state call followed by
    open_discovery: open_discovery itself only accepts seed queries, so it
    cannot zero the recipe on its own."""
    await client.call_tool(
        "open_discovery",
        {"initial_semantic_query": "trade", "initial_keyword_query": "africa"},
    )
    await client.call_tool("reset_state", {})
    result = await client.call_tool("open_discovery", {})
    data = result.structured_content
    assert data["semantic_query"] == ""
    assert data["keyword_query"] == ""
    assert data["committed_core_table"] == []
    assert data["last_saved_at"] is None


@pytest.mark.asyncio
async def test_reset_state_then_seed_resets_then_seeds(client) -> None:
    """Reset clears prior state, then a fresh open_discovery seed lands cleanly."""
    await client.call_tool(
        "open_discovery",
        {"initial_semantic_query": "old query", "initial_keyword_query": "old kw"},
    )
    await client.call_tool("reset_state", {})
    result = await client.call_tool(
        "open_discovery",
        {"initial_semantic_query": "fresh topic"},
    )
    data = result.structured_content
    assert data["semantic_query"] == "fresh topic"
    assert data["keyword_query"] == ""


@pytest.mark.asyncio
async def test_reset_state_zeros_recipe(client) -> None:
    await client.call_tool(
        "open_discovery",
        {"initial_semantic_query": "trade", "initial_keyword_query": "africa"},
    )
    reset_result = await client.call_tool("reset_state", {})
    data = reset_result.structured_content
    assert "cleared_at" in data
    assert data["reason"] == "user requested"

    # Subsequent open_discovery sees zeroed state.
    fresh = await client.call_tool("open_discovery", {})
    state = fresh.structured_content
    assert state == {
        "semantic_query": "",
        "keyword_query": "",
        "committed_core_table": [],
        "last_saved_at": None,
        "core_model_yaml": None,
        "last_validation_errors": None,
        "last_analysis": None,
        "core_model_certified_map": None,
    }


@pytest.mark.asyncio
async def test_download_session_returns_session_json_envelope(client) -> None:
    """download_session emits the substrate-defined session.json shape:
    ``{schema_version, generated_at, schema, yaml, queries}`` per
    osaa_metrics.session."""
    await client.call_tool(
        "open_discovery",
        {"initial_semantic_query": "trade", "initial_keyword_query": "africa"},
    )
    result = await client.call_tool("download_session", {})
    data = result.structured_content
    assert data["filename"].startswith("osaa-session-")
    assert data["filename"].endswith(".json")
    assert data["mime_type"] == "application/json"

    payload = json.loads(data["content"])
    assert payload["schema_version"] == "1"
    assert "generated_at" in payload
    assert payload["schema"] == []  # nothing committed yet
    assert payload["yaml"] is None
    assert payload["queries"] == []


@pytest.mark.asyncio
async def test_save_core_schema_then_build_core_table_invalidates_downstream(
    client, monkeypatch
) -> None:
    """``save_core_schema`` commits the core table schema and ``build_core_table``
    runs the pivot; after the pair the recipe carries the saved rows and no
    core model YAML or stored analysis, and the handle stays server-side. The
    pivot is monkeypatched so this exercises the tool sequence, not DuckDB."""
    import polars as pl

    class _FakeTbl:
        @property
        def columns(self):
            return ["country", "year", "gdp"]

        def count(self):
            class _R:
                def execute(self_inner):
                    return 0

            return _R()

    monkeypatch.setattr(
        "osaa_metrics.mcp.tools.enrich_core_table",
        lambda rows, *, cache_df: pl.DataFrame(),
    )
    monkeypatch.setattr(
        "osaa_metrics.mcp.tools.build_wide_table",
        lambda enriched, **kw: _FakeTbl(),
    )
    # Patch the catalogue lookup on the DataSourceProvider class so the
    # server's per-instance provider returns the stubbed df.
    from osaa_metrics.mcp.providers import DataSourceProvider

    monkeypatch.setattr(
        DataSourceProvider, "get_catalogue", lambda self: pl.DataFrame()
    )

    rows = [
        {"code": "NY.GDP.MKTP.CD", "var_name": "gdp", "description": "GDP"},
    ]
    save_result = await client.call_tool("save_core_schema", {"working_set": rows})
    save_data = save_result.structured_content
    assert save_data["row_count"] == 1
    assert save_data["schema"][0]["code"] == "NY.GDP.MKTP.CD"

    build_result = await client.call_tool("build_core_table", {})
    build_data = build_result.structured_content
    assert build_data["built"] is True
    assert "core_tbl_handle" not in build_data

    state = (await client.call_tool("open_discovery", {})).structured_content
    assert state["committed_core_table"][0]["code"] == "NY.GDP.MKTP.CD"
    assert state["last_saved_at"] == save_data["saved_at"]
    assert state["core_model_yaml"] is None
    assert state["last_analysis"] is None


@pytest.mark.asyncio
async def test_build_reports_model_invalidated_when_certified_map_already_gone(
    ctx, monkeypatch
) -> None:
    """model_invalidated must reflect whether a core semantic table existed
    before this build's wipe ran, not whether the certified map happened to
    survive to this call. Construct that split directly: a core-model YAML
    still sits on the recipe while its certified map is already gone, so
    ``breaking`` is forced True by the ``certified is None`` branch alone —
    the certified-map proxy would misreport nothing was lost."""
    import polars as pl

    from osaa_metrics.mcp.tools import handle_build_core_table

    class _FakeTbl:
        @property
        def columns(self):
            return ["country", "year", "gdp"]

        def count(self):
            class _R:
                def execute(self_inner):
                    return 0

            return _R()

    monkeypatch.setattr(
        "osaa_metrics.mcp.tools.enrich_core_table",
        lambda rows, *, cache_df: pl.DataFrame(),
    )
    monkeypatch.setattr(
        "osaa_metrics.mcp.tools.build_wide_table",
        lambda enriched, **kw: _FakeTbl(),
    )

    rows = [{"code": "NY.GDP.MKTP.CD", "var_name": "gdp", "description": "GDP"}]
    await session_state.save_core_schema(ctx, rows, "2026-05-04T10:00:00+00:00")
    recipe = await session_state.get_recipe(ctx)
    recipe.core_model_yaml = "yaml-text"
    await session_state._save_recipe(ctx, recipe)

    result = await handle_build_core_table(ctx, None, pl.DataFrame(), {})

    assert result["model_invalidated"] is True


# =========================================================================
# Session UUID stability probe — pins FastMCP's session-keying contract
# =========================================================================


@pytest.mark.asyncio
async def test_session_id_stable_within_one_client_session() -> None:
    """Two tool calls in one Client share a session_id, and two separate Client
    blocks get different ones — the keying every per-session read and write in
    session_state depends on."""
    import ibis
    import numpy as np

    from osaa_metrics.config import load_settings
    from osaa_metrics.mcp.providers import DataSourceProvider, EncoderProvider
    from osaa_metrics.mcp.server import OSAAMetricsServer

    class _StubEnc:
        def encode(self, sentences, *, normalize_embeddings: bool = True):
            return np.zeros((len(sentences), 1024), dtype="float32")

    mock_con = ibis.duckdb.connect(":memory:")
    server = OSAAMetricsServer(
        data_source=DataSourceProvider(
            load_settings(), con_factory=lambda settings: mock_con
        ),
        encoder_provider=EncoderProvider(factory=lambda: _StubEnc()),
        models={},
    )

    async with Client(server) as client_a:
        s1 = (await client_a.call_tool("open_discovery", {})).structured_content
        # Force a state mutation, then re-read.
        await client_a.call_tool("open_discovery", {"initial_semantic_query": "marker"})
        s2 = (await client_a.call_tool("open_discovery", {})).structured_content
        # If sessions stayed the same, the marker persists.
        assert s2["semantic_query"] == "marker"
        # The implicit assertion: get_state returned by call_tool was keyed
        # consistently, so the second call's state reflected the first call's
        # mutation. A session_id rotating mid-Client would fail this.
        _ = s1

    async with Client(server) as client_b:
        # Fresh client = fresh session = no marker carry-over.
        fresh = (await client_b.call_tool("open_discovery", {})).structured_content
        assert fresh["semantic_query"] == ""


@pytest.mark.asyncio
async def test_save_core_schema_invalidates_downstream(ctx) -> None:
    from osaa_metrics.mcp import session_state
    from osaa_metrics.mcp_discovery.tools import handle_save_core_schema

    # Seed downstream state as if a model + query already happened.
    cache = session_state.get_cache(ctx)
    cache.core_semantic_table = object()
    await session_state.append_query_chart_pair(
        ctx,
        dimensions=["region"],
        measures=["m"],
        filters=None,
        order_by=None,
        limit=None,
        chart_spec={"mark": "bar"},
        sql="SELECT 1",
        label="seed",
    )

    models = {f"{ctx.session_id}:seeded": object(), "other-session:keep": object()}
    await handle_save_core_schema(
        ctx, [{"code": "NY.GDP.MKTP.CD", "var_name": "gdp"}], models=models
    )

    cache = session_state.get_cache(ctx)
    recipe = await session_state.get_recipe(ctx)
    assert cache.core_semantic_table is None
    assert recipe.last_analysis is None
    assert list(models) == ["other-session:keep"]
    assert recipe.core_model_yaml is None


@pytest.mark.asyncio
async def test_append_query_chart_pair_stores_label(ctx) -> None:
    from osaa_metrics.mcp import session_state

    await session_state.append_query_chart_pair(
        ctx,
        dimensions=["region"],
        measures=["m"],
        filters=None,
        order_by=None,
        limit=None,
        chart_spec={"mark": "bar"},
        sql="SELECT 1",
        label="Debt by region",
    )

    recipe = await session_state.get_recipe(ctx)
    pair = recipe.last_analysis["queries"][0]
    assert pair["label"] == "Debt by region"


def test_cache_cap_at_least_headcount() -> None:
    """The LRU kitchen cap must hold a full workshop room of concurrent
    sessions, so nobody's core table handle is evicted mid-session."""
    from osaa_metrics.mcp import session_state

    assert session_state._CACHE_MAX_SESSIONS >= 16
