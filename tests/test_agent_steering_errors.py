"""Refusals steer: a tool that cannot run yet names, in its error, the tool
the agent should call to unblock itself. A load_session call with invalid JSON
is refused before anything mutates, so the session it lands in is unchanged."""

import json

import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import DATA_AVAILABLE, NO_DATA_REASON
from tests.test_invalidation_guard import WORKING_SET, _payload, _save_build_define

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATA_AVAILABLE, reason=NO_DATA_REASON),
]


async def test_build_before_save_refuses(client):
    with pytest.raises(ToolError):
        await client.call_tool("build_core_table", {})


async def test_define_before_build_steers_to_build(client):
    await client.call_tool("save_core_schema", {"working_set": WORKING_SET})
    with pytest.raises(ToolError, match="build_core_table") as e:
        await client.call_tool(
            "define_semantic_model", {"yaml_text": "m:\n  table: core_tbl\n"}
        )
    assert "build_core_table" in str(e.value)
    assert "save_core_schema" not in str(e.value)


async def test_summary_before_build_steers_to_build(client):
    with pytest.raises(ToolError, match="core table not built yet") as e:
        await client.call_tool("build_core_summary", {})
    assert "core table not built yet" in str(e.value)


async def test_summary_after_save_before_build_steers_to_build(client):
    await client.call_tool("save_core_schema", {"working_set": WORKING_SET})
    with pytest.raises(ToolError, match="core table not built yet") as e:
        await client.call_tool("build_core_summary", {})
    assert "build_core_table" in str(e.value)
    assert "save_core_schema" not in str(e.value)


async def test_query_unknown_model_steers_to_define(client):
    with pytest.raises(ToolError, match="unknown model") as e:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": "nope",
                "measures": ["x"],
                "chart_spec": {"mark": "bar"},
                "label": "l",
            },
        )
    assert "unknown model" in str(e.value)
    assert "define_semantic_model" in str(e.value)


async def test_load_session_invalid_json_mutates_nothing(client):
    await _save_build_define(client)
    before = _payload(await client.call_tool("download_session", {}))["content"]
    with pytest.raises(ToolError, match="session validation failed"):
        await client.call_tool("load_session", {"json_text": "not json at all"})
    after = _payload(await client.call_tool("download_session", {}))["content"]
    b, a = json.loads(before), json.loads(after)
    assert (b["schema"], b["yaml"], b["queries"]) == (
        a["schema"],
        a["yaml"],
        a["queries"],
    )
