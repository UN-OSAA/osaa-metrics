"""Validation tests for ``osaa_metrics.semantic.validate_and_load_model`` — the
substrate gate that the ``define_semantic_model`` MCP tool wraps. Tests the
four phases (JSON Schema, BSL from_config, expression invocation against the
live core_tbl, strict-loader duplicate-name guard) directly against the
substrate, plus client-level tests that exercise the full MCP flow.
"""

from __future__ import annotations

import os

import polars as pl
import pytest
import yaml
from fastmcp.exceptions import ToolError

from osaa_metrics.core_table import build_wide_table
from osaa_metrics.semantic import ModelValidationError, validate_and_load_model

needs_net = pytest.mark.skipif(
    os.environ.get("OSAA_SKIP_NET") == "1",
    reason="network-backed test skipped via OSAA_SKIP_NET=1",
)


def _get_con():
    """Settings-driven connection for substrate-level tests. Routes through
    the local ``data/`` mirror by default (conftest sets OSAA_DATA_MASTER_URL
    / OSAA_DATA_META_URL when present); export those vars to run against
    another source."""
    from osaa_metrics.config import build_connection, load_settings

    return build_connection(load_settings())


# Indicator codes the ``master`` table carries.
GDP = "NY.GDP.MKTP.CD"
GDP_GROWTH = "NY.GDP.MKTP.KD.ZG"


def _materialize_core_tbl(rows: list[dict]):
    """Build a core_tbl handle directly. Tests pass ``source``/``database``
    explicitly so we skip the catalogue enrichment that ``handle_build_core_table``
    does in production."""
    df = pl.DataFrame(
        [
            {
                "source": r["source"],
                "database": r["database"],
                "indicator_code": r["indicator_code"],
                "var_name": r["var_name"],
            }
            for r in rows
        ]
    )
    return build_wide_table(df, con=_get_con())


def _parse_validation_error(exc: ToolError) -> list[dict]:
    """Extract the structured errors list from a ``define_semantic_model`` ToolError.

    Contract: message body is ``"semantic model validation failed: <json>"``
    where <json> is ``{"errors": [{"path", "message", "hint"}, ...]}``.
    """
    import json

    msg = str(exc)
    assert msg.startswith("semantic model validation failed:"), (
        f"unexpected error prefix: {msg!r}"
    )
    payload = json.loads(msg[msg.index("{") :])
    return payload["errors"]


# --- Substrate-level validation phases ---


@needs_net
def test_missing_table_field_returns_schema_error():
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = """
my_model:
  measures:
    total_gdp: _.gdp.sum()
"""
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl)
    paths = [e["path"] for e in exc_info.value.errors]
    assert any("/my_model" in p for p in paths)


@needs_net
def test_unknown_column_returns_phase_c_attribute_error():
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = """
my_model:
  table: core_tbl
  measures:
    total_nonsense:
      expr: _.nonexistent_col.sum()
"""
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl)
    errors = exc_info.value.errors
    msgs = [e["message"] for e in errors]
    assert any("nonexistent_col" in m or "AttributeError" in m for m in msgs)
    hints = [e["hint"] for e in errors]
    assert any("var_names" in h or "core" in h for h in hints)


@needs_net
def test_malformed_expression_returns_safeval_error():
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = """
my_model:
  table: core_tbl
  measures:
    bad:
      expr: "_.gdp.mean("
"""
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl)
    errors = exc_info.value.errors
    # BSL's safe-eval fails to parse the malformed expression inside
    # from_config; the underlying SyntaxError is surfaced through the
    # normalized message.
    assert any("SyntaxError" in e["message"] for e in errors)


@needs_net
def test_duplicate_dimension_names_rejected_by_phase_d():
    """yaml.safe_load silently drops duplicates; our strict loader surfaces them."""
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = """
my_model:
  table: core_tbl
  dimensions:
    region: _.region
    region: _.country
  measures:
    total_gdp: _.gdp.sum()
"""
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl)
    assert any("duplicate" in e["message"].lower() for e in exc_info.value.errors)


@needs_net
def test_duplicate_measure_names_rejected_by_phase_d():
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = """
my_model:
  table: core_tbl
  measures:
    total_gdp: _.gdp.sum()
    total_gdp: _.gdp.mean()
"""
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl)
    assert any("duplicate" in e["message"].lower() for e in exc_info.value.errors)


# --- Phase A: YAML anchors / aliases refused (closes the billion-laughs gap
#     that the source-size cap can't catch) ---


def test_yaml_anchor_definition_rejected_in_phase_a():
    """Even a single ``&anchor`` in the source is refused — semantic models
    have no legitimate use for anchors, and accepting them opens the
    billion-laughs expansion DoS. The error fires during parsing, before
    Phase B (BSL) and before ``core_tbl`` is touched, so we can pass None."""
    yaml_text = """
my_model:
  table: core_tbl
  dimensions:
    region: &shared _.region
  measures:
    avg_gdp: _.gdp.mean()
"""
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl=None)
    assert any("anchor" in e["message"].lower() for e in exc_info.value.errors)


def test_yaml_alias_reference_rejected_in_phase_a():
    """A document that defines an anchor and references it via ``*`` is
    refused at the definition site — the error names the anchor."""
    yaml_text = """
my_model:
  table: core_tbl
  dimensions:
    region: &dim _.region
    country: *dim
  measures:
    avg_gdp: _.gdp.mean()
"""
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl=None)
    # The anchor is hit first (definition before reference) so the message
    # mentions "anchor 'dim'".
    assert any(
        "anchor" in e["message"].lower() and "dim" in e["message"]
        for e in exc_info.value.errors
    )


def test_billion_laughs_payload_rejected_without_expansion():
    """The textbook billion-laughs shape — a chain of anchors each doubling
    the previous — must be rejected during the FIRST anchor definition, long
    before any expansion would happen. Sanity: a real expansion would blow
    memory; this test runs in milliseconds because we stop at the first ``&``."""
    yaml_text = (
        "my_model:\n"
        "  table: core_tbl\n"
        "  lol1: &lol1 [lol, lol, lol, lol, lol, lol, lol, lol]\n"
        "  lol2: &lol2 [*lol1, *lol1, *lol1, *lol1, *lol1, *lol1, *lol1, *lol1]\n"
        "  lol3: &lol3 [*lol2, *lol2, *lol2, *lol2, *lol2, *lol2, *lol2, *lol2]\n"
        "  measures:\n"
        "    x: _.gdp.sum()"
    )
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl=None)
    # The first &lol1 is the trip-wire.
    assert any("anchor" in e["message"].lower() for e in exc_info.value.errors)


def test_anchor_free_yaml_still_loads_through_phase_a():
    """A normal anchor-free YAML still passes Phase A: it may fail at Phase B
    against ``core_tbl=None``, but the failure must not be the anchor refusal."""
    yaml_text = """
my_model:
  table: core_tbl
  dimensions:
    region: _.region
  measures:
    avg_gdp: _.gdp.mean()
"""
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl=None)
    # Whatever Phase B says, the error MUST NOT be the anchor-refusal one.
    assert not any("anchor" in e["message"].lower() for e in exc_info.value.errors)
    assert not any("alias" in e["message"].lower() for e in exc_info.value.errors)


# --- expr-string safety: hostile expressions rejected by BSL's safe-eval ---


@pytest.mark.parametrize("body", ["just a string", ["a", "b"], 5])
def test_non_dict_model_body_raises_clean_error(body):
    """A truthy non-dict model body (string/list/int) must surface as a clean
    ``ModelValidationError`` — the JSON Schema's own "not of type 'object'"
    error on the model definition — not a bare exception."""
    yaml_text = yaml.safe_dump({"econ": body})
    with pytest.raises(ModelValidationError):
        validate_and_load_model(yaml_text, core_tbl=None)


def test_deeply_nested_unary_expr_raises_clean_error():
    """A run of stacked unary minuses is a non-empty string, so it passes the
    schema and reaches ``ast.parse`` inside BSL's safe-eval during
    ``from_config``. Nested this deep, the C parser's own stack-depth guard
    trips and raises something other than ``SyntaxError``; that exception must
    not escape ``validate_and_load_model`` raw — the ``except Exception`` around
    ``from_config`` has to catch it and surface a clean ``ModelValidationError``."""
    deep_unary = "-" * 40000 + "1"
    yaml_text = f'econ:\n  table: core_tbl\n  measures:\n    bad: "{deep_unary}"\n'
    with pytest.raises(ModelValidationError):
        validate_and_load_model(yaml_text, core_tbl=None)


@needs_net
def test_backend_reaching_expr_rejected():
    """``_._find_backend().raw_sql(...)`` reaches for the ibis backend to run
    raw SQL. BSL's safe-eval rejects it during ``from_config`` — the private
    attribute and the ``raw_sql`` method are both outside its allowlist — and
    the ``SafeEvalError`` flows through ``_normalize_bsl_exception`` as a clean
    ``ModelValidationError``."""
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = (
        "econ:\n"
        "  table: core_tbl\n"
        "  dimensions:\n"
        "    bad: _._find_backend().raw_sql('select 1')\n"
        "  measures:\n"
        "    total: _.gdp.sum()\n"
    )
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl)
    errors = exc_info.value.errors
    assert any("SafeEvalError" in e["message"] for e in errors)


@needs_net
def test_fullwidth_unicode_dunder_rejected():
    """A dunder disguised with fullwidth underscores must still be rejected.
    CPython's parser NFKC-folds identifiers to ASCII before BSL's safe-eval
    inspects the AST, so the folded ``__class__`` trips the private-attribute
    rule inside ``from_config`` — no home-grown NFKC pass required."""
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = (
        "econ:\n"
        "  table: core_tbl\n"
        "  dimensions:\n"
        '    bad: "_.＿＿class＿＿"\n'
        "  measures:\n"
        "    total: _.gdp.sum()\n"
    )
    with pytest.raises(ModelValidationError):
        validate_and_load_model(yaml_text, core_tbl)


@needs_net
def test_non_allowlisted_method_rejected():
    """``pipe`` is not in BSL's ``SAFE_METHOD_CALLS``; ``pipe(print)`` would hand
    the expression to an arbitrary callable. BSL's safe-eval rejects the
    non-allowlisted method call during ``from_config``."""
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = "econ:\n  table: core_tbl\n  measures:\n    sneaky: _.gdp.pipe(print)\n"
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl)
    assert any("SafeEvalError" in e["message"] for e in exc_info.value.errors)


@needs_net
def test_lambda_expression_rejected_by_bsl():
    """A lambda is a non-empty string, so it clears the structural schema and
    reaches BSL's safe-eval in from_config. BSL rejects ``lambda _: ...`` — the
    private argument name ``_`` is disallowed — as a clean ModelValidationError."""
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = (
        "econ:\n"
        "  table: core_tbl\n"
        "  measures:\n"
        "    m:\n"
        '      expr: "lambda _: _.gdp.sum()"\n'
    )
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl)
    assert any("SafeEvalError" in e["message"] for e in exc_info.value.errors)


@needs_net
def test_example_yaml_expressions_still_load():
    """The committed example grammar (columns, sum/mean/nunique, arithmetic,
    parens) must pass the allowlist untouched."""
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "exports_gdp",
            },
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP_GROWTH,
                "var_name": "imports_gdp",
            },
        ]
    )
    yaml_text = (
        "econ:\n"
        "  table: core_tbl\n"
        "  dimensions:\n"
        "    year: _.year\n"
        "  measures:\n"
        '    balance: "_.exports_gdp.mean() - _.imports_gdp.mean()"\n'
        "    n: _.imports_gdp.nunique()\n"
    )
    _, _, _, name = validate_and_load_model(yaml_text, core_tbl)
    assert name == "econ"


@needs_net
def test_over_restrictive_methods_now_load():
    """``std``, ``var``, and ``abs`` live in BSL's own ``SAFE_METHOD_CALLS``, so
    a model using them loads: the method allowlist is BSL's, and nothing
    narrower stands in front of it turning legitimate models away."""
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = (
        "econ:\n"
        "  table: core_tbl\n"
        "  measures:\n"
        "    spread: _.gdp.std()\n"
        "    v: _.gdp.var()\n"
        "    a: _.gdp.abs().sum()\n"
    )
    _, _, _, name = validate_and_load_model(yaml_text, core_tbl)
    assert name == "econ"


def test_schema_rejects_calculated_measures_key():
    """``calculated_measures`` and ``filter`` carry expressions that Phase C
    never invokes — it walks dimensions and measures only. The schema's
    ``additionalProperties: false`` on the model body is what keeps them out,
    so a key added to the schema without matching coverage in
    ``validate_and_load_model`` fails here first."""
    yaml_text = (
        "econ:\n"
        "  table: core_tbl\n"
        "  measures:\n"
        "    total: _.gdp.sum()\n"
        "  calculated_measures:\n"
        "    doubled: _.total * 2\n"
    )
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl=None)
    assert any("calculated_measures" in e["message"] for e in exc_info.value.errors)


def test_schema_rejects_filter_key():
    """See ``test_schema_rejects_calculated_measures_key`` — same guard for
    the ``filter`` key."""
    yaml_text = (
        "econ:\n"
        "  table: core_tbl\n"
        "  measures:\n"
        "    total: _.gdp.sum()\n"
        "  filter:\n"
        "    - _.gdp > 0\n"
    )
    with pytest.raises(ModelValidationError) as exc_info:
        validate_and_load_model(yaml_text, core_tbl=None)
    assert any("filter" in e["message"] for e in exc_info.value.errors)


# --- Client-level integration (full MCP flow) ---


@needs_net
@pytest.mark.asyncio
async def test_build_core_table_invalidates_prior_model(client) -> None:
    """A Save that re-points the certified ``(var_name -> code)`` pairs makes
    the next build breaking: the core-model YAML is dropped rather than carried
    onto the new core table, and ``define_semantic_model`` must run again."""
    rows1 = [{"code": GDP, "var_name": "gdp", "description": "GDP"}]
    await client.call_tool("save_core_schema", {"working_set": rows1})
    await client.call_tool("build_core_table", {})

    yaml_text = """
my_model:
  table: core_tbl
  measures:
    total_gdp: _.gdp.sum()
"""
    await client.call_tool("define_semantic_model", {"yaml_text": yaml_text})
    state = (await client.call_tool("open_discovery", {})).structured_content
    assert state["core_model_yaml"] == yaml_text

    # Re-pivot with a different schema → prior YAML is dropped.
    rows2 = [
        {"code": GDP_GROWTH, "var_name": "gdp_growth", "description": "GDP growth"}
    ]
    await client.call_tool("save_core_schema", {"working_set": rows2})
    await client.call_tool("build_core_table", {})

    state = (await client.call_tool("open_discovery", {})).structured_content
    assert state["core_model_yaml"] is None


@needs_net
@pytest.mark.asyncio
async def test_define_semantic_model_failure_envelope_is_error_through_client(
    client,
) -> None:
    """Wire contract: on YAML validation failure, the MCP envelope is
    ``isError=true`` and the message body is
    ``semantic model validation failed: <json>`` with a parseable JSON tail.
    The envelope label must tell the truth about failure — a validation refusal
    is never dressed up as a successful result."""
    initial_rows = [{"code": GDP, "var_name": "gdp", "description": "GDP"}]
    await client.call_tool("save_core_schema", {"working_set": initial_rows})
    await client.call_tool("build_core_table", {})

    bad_yaml = """
my_model:
  table: core_tbl
  measures:
    total_nonsense:
      expr: _.no_such_column.sum()
"""
    with pytest.raises(ToolError) as exc_info:
        await client.call_tool("define_semantic_model", {"yaml_text": bad_yaml})

    errors = _parse_validation_error(exc_info.value)
    assert errors, "errors list must be non-empty"
    for entry in errors:
        assert set(entry) >= {"path", "message", "hint"}


# --- Model name flows through from the YAML's top-level key ---


@needs_net
def test_validate_and_load_model_returns_yaml_top_level_key_as_model_name():
    """``validate_and_load_model`` returns the YAML's top-level mapping key as
    the 4th tuple element. BSL convention: a semantic model's name IS the
    top-level dict key in ``from_yaml`` output."""
    core_tbl = _materialize_core_tbl(
        [
            {
                "source": "wb",
                "database": "wdi",
                "indicator_code": GDP,
                "var_name": "gdp",
            },
        ]
    )
    yaml_text = """
rural_education_parity:
  table: core_tbl
  measures:
    total_gdp: _.gdp.sum()
"""
    _, _, _, model_name = validate_and_load_model(yaml_text, core_tbl)
    assert model_name == "rural_education_parity"


@needs_net
@pytest.mark.asyncio
async def test_define_semantic_model_registers_under_yaml_declared_name(client) -> None:
    """End-to-end: a YAML keyed ``econ:`` registers under a session-namespaced
    key in the shared ``self.models`` dict; ``query_and_chart`` resolves it by
    the key returned from ``define_semantic_model``; BSL's mounted
    ``list_models`` reports the same namespaced key."""
    rows = [{"code": GDP, "var_name": "gdp", "description": "GDP"}]
    await client.call_tool("save_core_schema", {"working_set": rows})
    await client.call_tool("build_core_table", {})

    yaml_text = """
econ:
  table: core_tbl
  dimensions:
    iso3: _.iso3
  measures:
    total_gdp: _.gdp.sum()
"""
    define_result = await client.call_tool(
        "define_semantic_model", {"yaml_text": yaml_text}
    )
    model_name = define_result.structured_content["model_name"]
    assert model_name.endswith(":econ")

    # BSL's mounted list_models sees the same namespaced key — the dict is shared.
    list_result = await client.call_tool("list_models", {})
    assert model_name in list_result.structured_content

    # query_and_chart resolves the model by the returned namespaced key.
    query_result = await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "label": "GDP by country",
            "dimensions": ["iso3"],
            "measures": ["total_gdp"],
            "limit": 5,
            "chart_spec": {
                "mark": "bar",
                "encoding": {
                    "x": {"field": "iso3", "type": "nominal"},
                    "y": {"field": "total_gdp", "type": "quantitative"},
                },
            },
        },
    )
    assert "vega_lite_spec" in query_result.structured_content


@pytest.mark.asyncio
async def test_define_semantic_model_replaces_previous_core_semantic_table(
    client,
) -> None:
    """One core semantic table per session: a second define replaces the first.
    The earlier name is gone from the registry and refused by query_and_chart."""
    rows = [{"code": GDP, "var_name": "gdp", "description": "GDP"}]
    await client.call_tool("save_core_schema", {"working_set": rows})
    await client.call_tool("build_core_table", {})

    yaml_econ = """
econ:
  table: core_tbl
  measures:
    total_gdp: _.gdp.sum()
"""
    yaml_growth = """
growth:
  table: core_tbl
  measures:
    total_gdp: _.gdp.sum()
"""
    res_econ = await client.call_tool("define_semantic_model", {"yaml_text": yaml_econ})
    res_growth = await client.call_tool(
        "define_semantic_model", {"yaml_text": yaml_growth}
    )
    key_econ = res_econ.structured_content["model_name"]
    key_growth = res_growth.structured_content["model_name"]

    list_result = await client.call_tool("list_models", {})
    listed = set(list_result.structured_content)
    assert key_growth in listed
    assert key_econ not in listed

    with pytest.raises(ToolError) as exc_info:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": key_econ,
                "label": "GDP by country",
                "dimensions": ["iso3"],
                "measures": ["total_gdp"],
                "limit": 5,
                "chart_spec": {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "iso3", "type": "nominal"},
                        "y": {"field": "total_gdp", "type": "quantitative"},
                    },
                },
            },
        )
    assert "unknown model" in str(exc_info.value)

    query_result = await client.call_tool(
        "query_and_chart",
        {
            "model_name": key_growth,
            "label": "total GDP",
            "measures": ["total_gdp"],
            "chart_spec": {"mark": "bar"},
        },
    )
    assert "vega_lite_spec" in query_result.structured_content
