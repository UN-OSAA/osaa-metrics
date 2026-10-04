"""load_session's refusals: unknown indicator codes surface as a structured
tool error, not a raw traceback; YAML without a schema is a named refusal,
not a half-restore; an oversize embedded YAML is rejected before the PIVOT
runs; and the JSON-parse wrapper keeps the exception chain."""

import json

import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import DATA_AVAILABLE, NO_DATA_REASON
from tests.test_invalidation_guard import (
    THIRD_ROW,
    WORKING_SET,
    _payload,
    _save_build_define,
)

pytestmark = pytest.mark.skipif(not DATA_AVAILABLE, reason=NO_DATA_REASON)


def _session_doc(**overrides):
    doc = {
        "schema_version": "1",
        "generated_at": "2026-08-26T00:00:00+00:00",
        "schema": [dict(r) for r in WORKING_SET],
        "yaml": None,
        "queries": [],
    }
    doc.update(overrides)
    return doc


@pytest.mark.asyncio
async def test_unknown_codes_raise_structured_error_not_traceback(client):
    doc = _session_doc(
        schema=[{"code": "NOT-A-REAL-CODE", "description": "d", "var_name": "ghost"}]
    )
    with pytest.raises(ToolError, match="unknown indicator codes") as e:
        await client.call_tool("load_session", {"json_text": json.dumps(doc)})
    msg = str(e.value)
    assert "unknown indicator codes" in msg
    assert "Traceback" not in msg
    prefix = "core table schema invalid: "
    assert prefix in msg
    envelope = json.loads(msg[msg.index(prefix) + len(prefix) :])
    assert len(envelope["errors"]) == 1
    err = envelope["errors"][0]
    assert set(err) == {"path", "message", "hint"}
    assert "NOT-A-REAL-CODE" in err["message"]
    assert err["hint"]


@pytest.mark.asyncio
async def test_yaml_without_schema_is_named_refusal(client):
    doc = _session_doc(schema=[], yaml="m:\n  table: core_tbl\n")
    with pytest.raises(
        ToolError, match="session yaml requires a core table schema"
    ) as e:
        await client.call_tool("load_session", {"json_text": json.dumps(doc)})
    assert "session yaml requires a core table schema" in str(e.value)


@pytest.mark.asyncio
async def test_yaml_without_schema_mutates_nothing(client):
    await _save_build_define(client)
    before = _payload(await client.call_tool("download_session", {}))["content"]
    doc = _session_doc(schema=[], yaml="m:\n  table: core_tbl\n")
    with pytest.raises(ToolError, match="session yaml requires a core table schema"):
        await client.call_tool("load_session", {"json_text": json.dumps(doc)})
    after = _payload(await client.call_tool("download_session", {}))["content"]
    assert json.loads(before)["schema"] == json.loads(after)["schema"]


@pytest.mark.asyncio
async def test_oversize_session_yaml_is_rejected_before_pivot(client):
    from osaa_metrics.mcp.tools import CORE_MODEL_YAML_MAX

    model_name = await _save_build_define(client)
    # THIRD_ROW's var_name (ind_three) is not one of the columns the current
    # core table was pivoted on (ind_one, ind_two) — if the oversize check
    # fired after the PIVOT, this schema would already have replaced the
    # shared core table by the time the size check raises.
    oversized_yaml = "m:\n  table: core_tbl\n# " + ("x" * (CORE_MODEL_YAML_MAX + 1))
    doc = _session_doc(schema=[dict(THIRD_ROW)], yaml=oversized_yaml)
    with pytest.raises(ToolError, match="exceeds") as e:
        await client.call_tool("load_session", {"json_text": json.dumps(doc)})
    assert "exceeds" in str(e.value)

    # The PIVOT never ran: the model from _save_build_define still answers
    # against the untouched shared core table.
    result = await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "measures": ["avg_one"],
            "chart_spec": {"mark": "bar"},
            "label": "still alive",
        },
    )
    assert result is not None


def test_json_parse_error_keeps_chain():  # unit, no client needed
    from osaa_metrics.session import SessionValidationError, load_session

    try:
        load_session("{broken")
    except SessionValidationError as e:
        assert isinstance(e.__cause__, json.JSONDecodeError)
    else:
        raise AssertionError("expected SessionValidationError")
