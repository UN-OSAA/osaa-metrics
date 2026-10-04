"""Unit tests for MCP tool input hardening + discover_indicators logic + summary
formatter. Live tests use the local ``data/`` mirror by default —
skip with ``OSAA_SKIP_NET=1`` for offline runs."""

from __future__ import annotations

import os

import pytest

from osaa_metrics import summarize
from osaa_metrics.config import load_settings
from osaa_metrics.mcp import tools
from osaa_metrics.mcp.providers import DataSourceProvider
from osaa_metrics.mcp.tools import ToolValidationError

needs_net = pytest.mark.skipif(
    os.environ.get("OSAA_SKIP_NET") == "1",
    reason="network-backed test skipped via OSAA_SKIP_NET=1",
)


@pytest.fixture
def _live_con():
    """A settings-driven connection + catalogue df, threaded through providers
    so the discover_indicators wire shape under test matches what the MCP
    server actually wires."""
    provider = DataSourceProvider(load_settings())
    try:
        con = provider.get_con()
    except ToolValidationError as e:
        pytest.skip(f"data source unreachable: {e}")
    return con, provider.get_catalogue()


# --- input hardening ---


def test_harden_string_accepts_valid_var_name():
    tools._harden_string(
        "gdp_growth",
        "var_name",
        regex=tools._VAR_NAME_RE,
        regex_msg="must match ^[a-z_][a-z0-9_]*$",
    )  # should not raise


def test_harden_string_rejects_bad_var_names():
    for bad in ["GdP", "1abc", "abc-def", "a b", ""]:
        with pytest.raises(ToolValidationError):
            tools._harden_string(
                bad,
                "var_name",
                regex=tools._VAR_NAME_RE,
                regex_msg="must match ^[a-z_][a-z0-9_]*$",
            )


def test_harden_string_rejects_indicator_code_query_params():
    for bad in ["NY.GDP?fields=name", "NY.GDP#anchor"]:
        with pytest.raises(ToolValidationError):
            tools._harden_string(
                bad,
                "indicator_code",
                forbid=tools._INDICATOR_CODE_BAD,
                regex_msg="contains '?' or '#'",
            )


def test_harden_string_rejects_control_chars():
    with pytest.raises(ToolValidationError, match="control characters"):
        tools._harden_string("hello\x07world", "query")


def test_harden_string_allows_yaml_whitespace():
    """Multi-line core model YAML contains \\n / \\r / \\t — the control-character
    check must let those through, or define_semantic_model and load_session
    could not accept inline YAML."""
    yaml_text = "model:\n  table: core_tbl\n\tindented: ok\r\n"
    tools._harden_string(yaml_text, "core_model_yaml")  # should not raise


def test_validate_core_table_caps_at_100_rows():
    rows = [{"code": "X", "var_name": f"v_{i}"} for i in range(101)]
    with pytest.raises(ToolValidationError, match="capped at 100"):
        tools.validate_core_table_arg(rows)


def test_validate_core_table_rejects_duplicate_var_names():
    with pytest.raises(ToolValidationError, match="duplicate var_name"):
        tools.validate_core_table_arg(
            [
                {"code": "A", "var_name": "x"},
                {"code": "B", "var_name": "x"},
            ]
        )


def test_validate_core_table_keeps_description_from_widget_row_shape():
    """A raw discovery working-set row names the label ``meta_indicator_name``.

    The widget can Save those rows directly, so losing the label here is silent:
    the schema validates, and every downstream artifact carries a blank
    description while the widget's own Download (.csv) still shows it.
    """
    cleaned = tools.validate_core_table_arg(
        [
            {
                "indicator_code": "NY.GDP.MKTP.CD",
                "meta_indicator_name": "GDP (current US$)",
                "var_name": "gdp",
            },
        ]
    )
    assert cleaned[0]["description"] == "GDP (current US$)"
    assert cleaned[0]["code"] == "NY.GDP.MKTP.CD"


def test_validate_core_table_prefers_explicit_description():
    """The contract shape still wins when both keys are present."""
    cleaned = tools.validate_core_table_arg(
        [
            {
                "code": "A",
                "description": "explicit",
                "meta_indicator_name": "fallback",
                "var_name": "x",
            },
        ]
    )
    assert cleaned[0]["description"] == "explicit"


def test_validate_core_table_description_defaults_to_empty():
    """Neither key present is still legal — description is optional."""
    cleaned = tools.validate_core_table_arg([{"code": "A", "var_name": "x"}])
    assert cleaned[0]["description"] == ""


# --- discover_indicators logic ---

# Dummies for early-return / validation tests — never reached when
# semantic_query+keyword_query are both blank or top_k is invalid.
_NONE_DEPS = {"con": None, "encoder": None}


def test_discover_indicators_returns_empty_when_both_queries_blank():
    assert tools.discover_indicators(None, None, **_NONE_DEPS) == ([], False)
    assert tools.discover_indicators("", "  ", **_NONE_DEPS) == ([], False)


def test_discover_indicators_rejects_invalid_top_k():
    with pytest.raises(ToolValidationError):
        tools.discover_indicators("trade", None, top_k=0, **_NONE_DEPS)
    with pytest.raises(ToolValidationError):
        tools.discover_indicators("trade", None, top_k=-5, **_NONE_DEPS)


@needs_net
def test_discover_indicators_keyword_only_returns_real_rows(_live_con):
    """Keyword-only path uses the catalogue df, no encoder load.

    Verifies the wire shape (no description — it's filtered server-side and dropped
    from the projection) and that every returned row contains 'gdp' as a word in the
    visible columns.
    """
    con, _catalogue = _live_con
    rows, degraded = tools.discover_indicators(
        None,
        "gdp",
        top_k=10,
        con=con,
        encoder=None,
    )
    assert degraded is False
    assert isinstance(rows, list)
    assert len(rows) > 0
    assert "indicator_code" in rows[0]
    assert "source" in rows[0]
    assert "database" in rows[0]
    # Description is intentionally not part of the wire payload.
    assert "meta_indicator_description" not in rows[0]
    import re

    word = re.compile(r"\bgdp\b", re.IGNORECASE)
    for r in rows:
        haystack = (
            r.get("indicator_code", "") + " " + (r.get("meta_indicator_name") or "")
        )
        assert word.search(haystack), (
            f"row missing word 'gdp' in code+name: {r['indicator_code']}"
        )


@needs_net
def test_discover_indicators_keyword_uses_word_boundaries(_live_con):
    """The keyword filter matches whole words: 'LCU' must not match inside
    'caLCUlated' or 'incLUded', which would hand the analyst rows that never
    mention the unit.

    The keyword is matched against (code, name, description) server-side and
    description is stripped from the wire payload, so the assertion re-fetches
    it from the catalogue.
    """
    con, catalogue = _live_con
    rows, _ = tools.discover_indicators(
        "trade balance",
        "LCU",
        top_k=50,
        con=con,
        encoder=None,
    )
    import re

    word = re.compile(r"\blcu\b", re.IGNORECASE)
    desc_by_code = dict(
        zip(  # noqa: B905  # both columns come from one DataFrame
            catalogue["indicator_code"].to_list(),
            catalogue["meta_indicator_description"].to_list(),
        )
    )
    for r in rows:
        haystack = (
            r.get("indicator_code", "")
            + " "
            + (r.get("meta_indicator_name") or "")
            + " "
            + (desc_by_code.get(r["indicator_code"]) or "")
        )
        assert word.search(haystack), (
            f"row {r['indicator_code']} matched 'LCU' as a substring, not as a word"
        )


@needs_net
def test_discover_indicators_reports_degraded_on_encoder_failure(_live_con):
    """An encoder failure drops to the keyword fallback: rows still come back,
    flagged degraded=True, never as a tool error."""
    con, _ = _live_con

    class _Raising:
        def encode(self, sentences, *, normalize_embeddings: bool = True):
            raise RuntimeError("simulated encoder failure")

    rows, degraded = tools.discover_indicators(
        "gdp",
        None,
        top_k=5,
        con=con,
        encoder=_Raising(),
    )
    assert degraded is True
    assert isinstance(rows, list) and len(rows) > 0


# --- summary formatter ---


def test_fmt_int_thousands():
    assert summarize._fmt_int(12847) == "12,847"
    assert summarize._fmt_int(0) == "0"
    assert summarize._fmt_int(None) == ""


def test_fmt_num_g_and_exponent():
    assert summarize._fmt_num(0) == "0"
    assert summarize._fmt_num(None) == ""
    assert summarize._fmt_num(3.47) == "3.47"
    # Outside [1e-3, 1e5] → exponent form.
    assert "e" in summarize._fmt_num(1.4e10)
    assert "e" in summarize._fmt_num(0.0001)


def test_markdown_table_header_separator_and_alignment():
    md = summarize._markdown_table(
        headers=["a", "longer_col"],
        rows=[["1", "x"], ["22", "yyy"]],
    )
    lines = md.splitlines()
    assert lines[0].startswith("| a")
    assert "longer_col" in lines[0]
    assert lines[1].startswith("|")
    assert "-" in lines[1]
    assert lines[2].startswith("| 1")


# --- open_discovery seed hardening ---


@pytest.mark.asyncio
async def test_open_discovery_rejects_control_chars_in_semantic_seed(client) -> None:
    """A control character in `initial_semantic_query` must surface as a tool
    error rather than being silently persisted into the recipe."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await client.call_tool(
            "open_discovery",
            {"initial_semantic_query": "trade\x00balance"},
        )
    assert "initial_semantic_query" in str(exc_info.value)
    assert "control characters" in str(exc_info.value)


@pytest.mark.asyncio
async def test_open_discovery_accepts_clean_seeds_and_persists_them(client) -> None:
    """A normal-shape seed lands in `recipe.semantic_query` / `keyword_query`."""
    result = await client.call_tool(
        "open_discovery",
        {
            "initial_semantic_query": "trade balance",
            "initial_keyword_query": "africa",
        },
    )
    state = result.structured_content
    assert state["semantic_query"] == "trade balance"
    assert state["keyword_query"] == "africa"


# --- Input-hardening boundary caps ---


@pytest.mark.asyncio
async def test_open_discovery_rejects_oversize_semantic_seed(client) -> None:
    """initial_semantic_query is bounded by KEYWORD_QUERY_MAX (256 chars) for
    parity with discover_indicators. Larger seeds reject at the tool boundary."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await client.call_tool(
            "open_discovery",
            {"initial_semantic_query": "x" * 257},
        )
    assert "initial_semantic_query" in str(exc_info.value)
    assert "256-char cap" in str(exc_info.value)


@pytest.mark.asyncio
async def test_open_discovery_rejects_oversize_keyword_seed(client) -> None:
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await client.call_tool(
            "open_discovery",
            {"initial_keyword_query": "x" * 257},
        )
    assert "initial_keyword_query" in str(exc_info.value)


@pytest.mark.asyncio
async def test_load_session_rejects_oversize_json_text(client) -> None:
    """load_session caps the JSON envelope at SESSION_JSON_MAX (4 MiB).
    A larger payload rejects before json.loads runs."""
    from fastmcp.exceptions import ToolError

    oversized = "{" + " " * (4 * 1024 * 1024) + "}"  # 4 MiB + 2 chars
    with pytest.raises(ToolError) as exc_info:
        await client.call_tool("load_session", {"json_text": oversized})
    assert "json_text" in str(exc_info.value)


@pytest.mark.asyncio
async def test_query_and_chart_rejects_oversize_field_count(client) -> None:
    """dimensions + measures combined cap at QUERY_FIELD_MAX (64). Cap fires
    before the model-name lookup so this works without a registered model."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": "anything",
                "dimensions": [f"d{i}" for i in range(40)],
                "measures": [f"m{i}" for i in range(40)],
                "chart_spec": {"mark": "bar"},
                "label": "test",
            },
        )
    assert "exceeds 64-field cap" in str(exc_info.value)


@pytest.mark.asyncio
async def test_query_and_chart_rejects_oversize_filters_list(client) -> None:
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": "anything",
                "filters": [{"field": "x", "op": "=", "value": i} for i in range(40)],
                "chart_spec": {"mark": "bar"},
                "label": "test",
            },
        )
    assert "exceeds 32-entry cap" in str(exc_info.value)


@pytest.mark.asyncio
async def test_query_and_chart_rejects_string_filter(client) -> None:
    """A filter entry typed as text is rejected at the tool boundary —
    string filters would otherwise reach BSL's eval path. Fires before
    the model-name lookup so no registered model is needed."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": "anything",
                "filters": ["_.year > 2000"],
                "chart_spec": {"mark": "bar"},
                "label": "test",
            },
        )
    assert "filters" in str(exc_info.value)


@pytest.mark.asyncio
async def test_handle_query_and_chart_rejects_string_filter(ctx) -> None:
    """Handler-level guard: the per-entry dict check raises even for callers
    that bypass the wire schema."""
    with pytest.raises(ToolValidationError, match="must be a JSON object"):
        await tools.handle_query_and_chart(
            ctx,
            models={},
            model_name="m",
            dimensions=["year"],
            measures=[],
            filters=["_.year > 2000"],
            order_by=None,
            limit=None,
            chart_spec={"mark": "bar"},
            label="t",
        )


@pytest.mark.asyncio
async def test_reset_state_rejects_oversize_reason(client) -> None:
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc_info:
        await client.call_tool("reset_state", {"reason": "x" * 257})
    assert "reason" in str(exc_info.value)
    assert "256-char cap" in str(exc_info.value)


@pytest.mark.asyncio
async def test_reset_state_accepts_default_reason(client) -> None:
    """The default reason (\"user requested\", 14 chars) must pass the cap."""
    result = await client.call_tool("reset_state", {})
    assert result.structured_content["reason"] == "user requested"


@pytest.mark.asyncio
async def test_query_and_chart_requires_chart_spec(client) -> None:
    """Missing chart_spec rejects at the tool boundary."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": "anything",
                "dimensions": ["region"],
                "measures": ["m"],
                "label": "test",
            },
        )
    assert "chart_spec" in str(exc.value)


@pytest.mark.asyncio
async def test_query_and_chart_rejects_empty_chart_spec(client) -> None:
    """An empty chart_spec dict is rejected before any model lookup."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": "anything",
                "dimensions": ["region"],
                "measures": ["m"],
                "chart_spec": {},
                "label": "test",
            },
        )
    assert "chart_spec" in str(exc.value)


@pytest.mark.asyncio
async def test_query_and_chart_accepts_json_stringified_chart_spec(client) -> None:
    """A JSON-stringified chart_spec is parsed before the gate runs.

    The spec gate passes (parsed dict is non-empty), so the error comes from
    the unknown-model lookup — not from chart_spec being rejected as
    missing/empty/non-dict.
    """
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": "anything",
                "chart_spec": '{"mark": "bar"}',
                "label": "test",
            },
        )
    assert "chart_spec" not in str(exc.value)


@pytest.mark.asyncio
async def test_query_and_chart_rejects_invalid_json_string_as_chart_spec(
    client,
) -> None:
    """A non-JSON string passed as chart_spec is rejected: the validator passes
    it through unchanged and pydantic's dict-type constraint catches it."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError) as exc:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": "anything",
                "chart_spec": "not a json object",
                "label": "test",
            },
        )
    assert "chart_spec" in str(exc.value)


@pytest.mark.asyncio
async def test_save_core_schema_accepts_json_stringified_working_set(client) -> None:
    """A JSON-stringified working_set is parsed and committed successfully."""
    result = await client.call_tool(
        "save_core_schema",
        {
            "working_set": '[{"code":"NY.GDP.MKTP.CD","description":"GDP","var_name":"gdp"}]',
        },
    )
    state = result.structured_content
    assert state["row_count"] == 1
    assert state["schema"][0]["var_name"] == "gdp"


# --- session-namespaced model key: cross-session isolation ---

_ISOLATION_WORKING_SET = [
    {"code": "NY.GDP.MKTP.CD", "description": "GDP", "var_name": "gdp"}
]

_ISOLATION_YAML = """\
econ_trade:
  table: core_tbl
  measures:
    total_gdp: _.gdp.sum()
"""


async def _drive_flow_to_define(c) -> None:
    """Drive save_core_schema → build_core_table → define_semantic_model
    against a single client session. Uses a minimal working set + YAML keyed
    'econ_trade' — real substrate, no mocks."""
    await c.call_tool("save_core_schema", {"working_set": _ISOLATION_WORKING_SET})
    await c.call_tool("build_core_table", {})
    await c.call_tool("define_semantic_model", {"yaml_text": _ISOLATION_YAML})


@needs_net
@pytest.mark.asyncio
async def test_define_model_is_session_namespaced() -> None:
    """Two sessions defining the same-named YAML get distinct model keys —
    each session gets a distinct namespaced key in the process-global models dict."""
    from osaa_metrics.encoders.local import LocalEncoder
    from osaa_metrics.mcp.providers import EncoderProvider
    from osaa_metrics.mcp.server import OSAAMetricsServer

    settings = load_settings()
    models: dict = {}
    server = OSAAMetricsServer(
        data_source=DataSourceProvider(settings),
        encoder_provider=EncoderProvider(factory=LocalEncoder),
        models=models,
    )

    async def define_in_new_session() -> None:
        from fastmcp import Client

        async with Client(server) as c:
            await _drive_flow_to_define(c)

    await define_in_new_session()
    await define_in_new_session()

    econ_keys = [k for k in models if k.endswith(":econ_trade")]
    assert len(econ_keys) == 2, f"expected 2 isolated keys, got {sorted(models)}"


@needs_net
@pytest.mark.asyncio
async def test_load_session_model_key_is_session_namespaced() -> None:
    """Restoring a session that includes a YAML model registers the model under
    a session-namespaced key — each session gets a distinct namespaced key."""
    from osaa_metrics.encoders.local import LocalEncoder
    from osaa_metrics.mcp.providers import EncoderProvider
    from osaa_metrics.mcp.server import OSAAMetricsServer

    settings = load_settings()
    models: dict = {}
    server = OSAAMetricsServer(
        data_source=DataSourceProvider(settings),
        encoder_provider=EncoderProvider(factory=LocalEncoder),
        models=models,
    )

    # Build state + export a session blob in one client session.
    from fastmcp import Client

    async with Client(server) as c:
        await c.call_tool("save_core_schema", {"working_set": _ISOLATION_WORKING_SET})
        await c.call_tool("build_core_table", {})
        await c.call_tool("define_semantic_model", {"yaml_text": _ISOLATION_YAML})
        download_result = await c.call_tool("download_session", {})
        session_json = download_result.structured_content["content"]

    # Restore that blob in a fresh session and check the returned model key.
    async with Client(server) as c:
        load_result = await c.call_tool("load_session", {"json_text": session_json})
        result_model_name = load_result.structured_content["model_name"]

    assert ":" in result_model_name, (
        f"expected session-namespaced key, got {result_model_name!r}"
    )
    assert result_model_name.endswith(":econ_trade"), (
        f"expected key ending in ':econ_trade', got {result_model_name!r}"
    )


# --- prepare_download ---


@pytest.mark.asyncio
async def test_prepare_download_session_returns_fetchable_token(client) -> None:
    res = await client.call_tool("prepare_download", {"kind": "session"})
    data = res.structured_content
    assert data["filename"].endswith(".json")
    assert "/download/" in data["url"]
    token = data["url"].rsplit("/", 1)[-1]
    from osaa_metrics.mcp import tools

    blob = tools._DOWNLOAD_TOKENS[token]
    assert blob["mime_type"] == "application/json"
    assert blob["bytes"]


@pytest.mark.asyncio
async def test_prepare_download_rejects_bad_kind(client) -> None:
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        await client.call_tool("prepare_download", {"kind": "pdf"})


@needs_net
@pytest.mark.asyncio
async def test_prepare_download_yaml_returns_yaml_token() -> None:
    """Happy path: after driving save → build → define, prepare_download {kind:'yaml'}
    stashes the core-model YAML under a token and returns the correct metadata."""
    from fastmcp import Client

    from osaa_metrics.encoders.local import LocalEncoder
    from osaa_metrics.mcp import tools
    from osaa_metrics.mcp.providers import EncoderProvider
    from osaa_metrics.mcp.server import OSAAMetricsServer

    settings = load_settings()
    server = OSAAMetricsServer(
        data_source=DataSourceProvider(settings),
        encoder_provider=EncoderProvider(factory=LocalEncoder),
        models={},
    )
    async with Client(server) as c:
        await _drive_flow_to_define(c)
        res = await c.call_tool("prepare_download", {"kind": "yaml"})
    data = res.structured_content
    assert data["filename"].endswith(".yaml")
    assert "/download/" in data["url"]
    token = data["url"].rsplit("/", 1)[-1]
    blob = tools._DOWNLOAD_TOKENS[token]
    assert blob["mime_type"] == "application/x-yaml"
    assert blob["bytes"]
    assert blob["bytes"].decode("utf-8")  # valid UTF-8 YAML text


@pytest.mark.asyncio
async def test_prepare_download_yaml_absent_model_raises(client) -> None:
    """Absent path: a fresh session with no model defined raises ToolError."""
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        await client.call_tool("prepare_download", {"kind": "yaml"})
