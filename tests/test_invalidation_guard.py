"""Tool-boundary behaviour for conditional invalidation: a core table rebuild
or a Save that leaves every certified (var_name -> code) pair intact keeps
the core-model YAML, the core semantic table, and the analysis history; a
rebuild or Save that drops or re-points a certified pair wipes them and
reports it via the ``model_invalidated`` flag."""

import json

import pytest
from fastmcp.exceptions import ToolError

from osaa_metrics.config import load_settings
from osaa_metrics.mcp import session_state
from osaa_metrics.mcp.providers import DataSourceProvider
from osaa_metrics.mcp.tools import handle_build_core_table
from tests.conftest import DATA_AVAILABLE, NO_DATA_REASON

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATA_AVAILABLE, reason=NO_DATA_REASON),
]

# Indicator codes the local parquet mirror carries.
WORKING_SET = [
    {
        "code": "NY.GDP.MKTP.CD",
        "description": "GDP (current US$)",
        "var_name": "ind_one",
    },
    {"code": "NY.GDP.MKTP.KD.ZG", "description": "GDP growth", "var_name": "ind_two"},
]

# A third column for the additive cases — a second var_name over a code the
# working set already carries (duplicate codes are legal; duplicate var_names
# are not).
THIRD_ROW = {
    "code": "NY.GDP.MKTP.CD",
    "description": "GDP again",
    "var_name": "ind_three",
}

MODEL_YAML = """\
econ_trade:
  table: core_tbl
  dimensions:
    iso3: _.iso3
  measures:
    avg_one:
      expr: _.ind_one.mean()
"""


def _payload(result):
    # House convention for tool results: the payload is the structured content.
    return result.structured_content


async def _save_build_define(client) -> str:
    await client.call_tool("save_core_schema", {"working_set": WORKING_SET})
    await client.call_tool("build_core_table", {})
    result = await client.call_tool("define_semantic_model", {"yaml_text": MODEL_YAML})
    return _payload(result)["model_name"]


async def test_rebuild_same_schema_keeps_model(client):
    model_name = await _save_build_define(client)
    result = await client.call_tool("build_core_table", {})
    assert _payload(result)["model_invalidated"] is False
    # the model still answers:
    q = await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "measures": ["avg_one"],
            "chart_spec": {"mark": "bar"},
            "label": "still alive",
        },
    )
    assert q is not None


async def test_resave_identical_schema_keeps_model(client):
    await _save_build_define(client)
    result = await client.call_tool("save_core_schema", {"working_set": WORKING_SET})
    assert _payload(result)["model_invalidated"] is False


async def test_additive_save_keeps_model(client):
    await _save_build_define(client)
    wider = [*WORKING_SET, THIRD_ROW]
    result = await client.call_tool("save_core_schema", {"working_set": wider})
    assert _payload(result)["model_invalidated"] is False


async def test_narrowing_save_wipes_model(client):
    await _save_build_define(client)
    result = await client.call_tool(
        "save_core_schema", {"working_set": WORKING_SET[:1]}
    )
    assert _payload(result)["model_invalidated"] is True
    # the yaml is gone: download_session reflects the wiped recipe
    doc = json.loads(
        _payload(await client.call_tool("download_session", {}))["content"]
    )
    assert doc["yaml"] is None


async def test_repointed_var_name_wipes_model(client):
    await _save_build_define(client)
    swapped = [
        WORKING_SET[0],
        {"code": WORKING_SET[0]["code"], "description": "other", "var_name": "ind_two"},
    ]
    # ind_two now maps to a DIFFERENT code than certified -> breaking
    result = await client.call_tool("save_core_schema", {"working_set": swapped})
    assert _payload(result)["model_invalidated"] is True
    # the wipe fired, not just the flag: download_session reflects the cleared recipe
    doc = json.loads(
        _payload(await client.call_tool("download_session", {}))["content"]
    )
    assert doc["yaml"] is None


async def test_breaking_build_clears_registry(client):
    model_name = await _save_build_define(client)
    await client.call_tool("save_core_schema", {"working_set": WORKING_SET[:1]})
    await client.call_tool("build_core_table", {})
    with pytest.raises(ToolError, match="unknown model") as e:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": model_name,
                "measures": ["avg_one"],
                "chart_spec": {"mark": "bar"},
                "label": "should be gone",
            },
        )
    assert "unknown model" in str(e.value)


async def test_restore_then_rebuild_keeps_model(client):
    await _save_build_define(client)
    session_json = _payload(await client.call_tool("download_session", {}))["content"]
    await client.call_tool("reset_state", {"reason": "fresh start"})
    load = _payload(await client.call_tool("load_session", {"json_text": session_json}))
    model_name = load["model_name"]
    result = await client.call_tool("build_core_table", {})
    assert _payload(result)["model_invalidated"] is False
    q = await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "measures": ["avg_one"],
            "chart_spec": {"mark": "bar"},
            "label": "restored and rebuilt",
        },
    )
    assert q is not None


async def test_restore_history_without_yaml_then_narrowing_save_reports_no_model(
    client,
):
    model_name = await _save_build_define(client)
    await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "measures": ["avg_one"],
            "chart_spec": {"mark": "bar"},
            "label": "history entry",
        },
    )
    session_json = _payload(await client.call_tool("download_session", {}))["content"]
    import json as _json

    doc = _json.loads(session_json)
    doc["yaml"] = None  # a history-only session file
    await client.call_tool("reset_state", {"reason": "fresh start"})
    await client.call_tool("load_session", {"json_text": _json.dumps(doc)})
    # A narrowing save after restoring a history-only session: no core
    # semantic table ever existed in this session, so nothing was discarded.
    result = await client.call_tool(
        "save_core_schema", {"working_set": WORKING_SET[:1]}
    )
    assert _payload(result)["model_invalidated"] is False
    assert _payload(result)["queries_invalidated"] is True


async def test_restore_history_without_yaml_survives_rebuild(client):
    model_name = await _save_build_define(client)
    await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "measures": ["avg_one"],
            "chart_spec": {"mark": "bar"},
            "label": "history entry",
        },
    )
    session_json = _payload(await client.call_tool("download_session", {}))["content"]
    import json as _json

    doc = _json.loads(session_json)
    doc["yaml"] = None  # a history-only session file
    await client.call_tool("reset_state", {"reason": "fresh start"})
    await client.call_tool("load_session", {"json_text": _json.dumps(doc)})
    await client.call_tool("build_core_table", {})
    state_after = _payload(await client.call_tool("download_session", {}))
    restored = _json.loads(state_after["content"])
    assert len(restored["queries"]) == 1  # restored history survived


# A second core-model YAML whose model name differs from MODEL_YAML's, so a
# restore registers a new key alongside the first instead of overwriting it.
OTHER_MODEL_YAML = """\
other_model:
  table: core_tbl
  dimensions:
    iso3: _.iso3
  measures:
    avg_one:
      expr: _.ind_one.mean()
"""


async def test_restore_drops_previously_defined_models(client):
    # No reset_state here: a restore into a live session must drop that
    # session's own registry entries by itself.
    model_name = await _save_build_define(client)
    doc = json.loads(
        _payload(await client.call_tool("download_session", {}))["content"]
    )
    # Same var_name, different indicator code, differently-named model: the
    # restored core table would answer the earlier model's measure with
    # numbers from another indicator.
    doc["schema"] = [
        {
            "code": "NY.GDP.MKTP.KD.ZG",
            "description": "GDP growth",
            "var_name": "ind_one",
        }
    ]
    doc["yaml"] = OTHER_MODEL_YAML
    await client.call_tool("load_session", {"json_text": json.dumps(doc)})
    with pytest.raises(ToolError, match="unknown model") as e:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": model_name,
                "measures": ["avg_one"],
                "chart_spec": {"mark": "bar"},
                "label": "must not resolve after a restore",
            },
        )
    assert "unknown model" in str(e.value)


async def test_redefine_clears_the_analysis_history(client):
    model_name = await _save_build_define(client)
    await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "measures": ["avg_one"],
            "chart_spec": {"mark": "bar"},
            "label": "before the redefine",
        },
    )
    before = json.loads(
        _payload(await client.call_tool("download_session", {}))["content"]
    )
    assert len(before["queries"]) == 1

    await client.call_tool("define_semantic_model", {"yaml_text": MODEL_YAML})

    after = json.loads(
        _payload(await client.call_tool("download_session", {}))["content"]
    )
    assert after["queries"] == []


async def test_breaking_rebuild_clears_the_analysis_history(ctx):
    # The certified pairs are stamped from WORKING_SET, then the saved rows
    # re-point ind_one to another code without going through the Save tool, so
    # build_core_table itself is the step that finds the break.
    provider = DataSourceProvider(load_settings())
    await session_state.save_core_schema(ctx, WORKING_SET, "2026-10-02T00:00:00")
    await session_state.set_core_semantic_table(ctx, object(), MODEL_YAML)
    await session_state.append_query_chart_pair(
        ctx,
        dimensions=["iso3"],
        measures=["avg_one"],
        filters=None,
        order_by=None,
        limit=None,
        chart_spec={"mark": "bar"},
        sql="SELECT 1",
        label="before the rebuild",
    )
    repointed = [{**WORKING_SET[0], "code": "NY.GDP.MKTP.KD.ZG"}, WORKING_SET[1]]
    await session_state.save_core_schema(ctx, repointed, "2026-10-02T00:00:01")

    result = await handle_build_core_table(
        ctx, provider.get_con(), provider.get_catalogue(), models={}
    )

    assert result["queries_invalidated"] is True
    assert (await session_state.get_recipe(ctx)).last_analysis is None
