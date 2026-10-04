"""Reads check provenance before answering: the built core table must match
the committed core table schema, and the core semantic table must match the
core table it was defined from — except a core table widened by extra
columns, which stays queryable since the core semantic table never
references what it hasn't seen. All three guards steer with instructional
errors instead of returning stale or wrong numbers."""

import json

import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import DATA_AVAILABLE, NO_DATA_REASON
from tests.test_invalidation_guard import (
    MODEL_YAML,
    THIRD_ROW,
    WORKING_SET,
    _payload,
    _save_build_define,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATA_AVAILABLE, reason=NO_DATA_REASON),
]


async def test_summary_after_new_save_demands_rebuild(client):
    await _save_build_define(client)
    wider = [*WORKING_SET, THIRD_ROW]
    await client.call_tool("save_core_schema", {"working_set": wider})
    with pytest.raises(ToolError, match="core table is stale") as e:
        await client.call_tool("build_core_summary", {})
    assert "core table is stale" in str(e.value)


async def test_define_after_new_save_demands_rebuild(client):
    await _save_build_define(client)
    wider = [*WORKING_SET, THIRD_ROW]
    await client.call_tool("save_core_schema", {"working_set": wider})
    with pytest.raises(ToolError, match="core table is stale") as e:
        await client.call_tool("define_semantic_model", {"yaml_text": MODEL_YAML})
    assert "core table is stale" in str(e.value)


async def test_summary_with_fresh_handle_still_works(client):
    await _save_build_define(client)
    result = await client.call_tool("build_core_summary", {})
    assert "core table — shape:" in _payload(result)["markdown"]


async def test_query_after_breaking_save_is_refused(client):
    # A breaking Save discards the core semantic table and its registration
    # together, so the name stops resolving at the Save. The staleness guard
    # is not reached here — the unknown-model check answers first.
    model_name = await _save_build_define(client)
    result = await client.call_tool(
        "save_core_schema", {"working_set": WORKING_SET[:1]}
    )
    assert _payload(result)["model_invalidated"] is True
    with pytest.raises(ToolError, match="unknown model") as e:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": model_name,
                "measures": ["avg_one"],
                "chart_spec": {"mark": "bar"},
                "label": "should be refused",
            },
        )
    assert "unknown model" in str(e.value)


BAD_MODEL_YAML = """\
econ_trade:
  table: core_tbl
  measures:
    avg_missing:
      expr: _.no_such_column.mean()
"""


async def test_failed_restore_leaves_no_queryable_core_table(client):
    # A restore rebuilds the core table before it validates the core-model
    # YAML, and the rebuild replaces the one table the handle names. When the
    # YAML then fails, the bookkeeping swap never runs on its own, so the
    # except branch drops the stale handle *and* this session's registry
    # entries by hand — nothing may answer from that state, on either our own
    # query_and_chart or BSL's mounted query_model.
    model_name = await _save_build_define(client)
    doc = json.loads(
        _payload(await client.call_tool("download_session", {}))["content"]
    )
    doc["schema"] = [
        {
            "code": "NY.GDP.MKTP.KD.ZG",
            "description": "GDP growth",
            "var_name": "ind_one",
        },
        WORKING_SET[1],
    ]
    doc["yaml"] = BAD_MODEL_YAML
    with pytest.raises(ToolError, match="session yaml validation failed") as load_error:
        await client.call_tool("load_session", {"json_text": json.dumps(doc)})
    assert "session yaml validation failed" in str(load_error.value)

    with pytest.raises(ToolError, match="unknown model") as e:
        await client.call_tool(
            "query_and_chart",
            {
                "model_name": model_name,
                "measures": ["avg_one"],
                "chart_spec": {"mark": "bar"},
                "label": "must not answer after a failed restore",
            },
        )
    assert "unknown model" in str(e.value)

    with pytest.raises(ToolError, match="not found") as query_model_error:
        await client.call_tool(
            "query_model",
            {"model_name": model_name, "measures": ["avg_one"]},
        )
    assert "not found" in str(query_model_error.value)


async def test_breaking_save_drops_the_core_semantic_table_from_the_registry(client):
    # A breaking Save discards the core semantic table, so the name it was
    # registered under must stop resolving immediately — not at the next
    # build. BSL's mounted query surface reads that registry without a
    # provenance check, so the entry outliving the Save is the whole exposure.
    model_name = await _save_build_define(client)
    result = await client.call_tool(
        "save_core_schema", {"working_set": WORKING_SET[:1]}
    )
    assert _payload(result)["model_invalidated"] is True
    listed = _payload(await client.call_tool("list_models", {}))
    assert model_name not in listed
    with pytest.raises(ToolError, match="not found") as e:
        await client.call_tool(
            "query_model",
            {"model_name": model_name, "measures": ["avg_one"]},
        )
    assert "not found" in str(e.value)


async def test_query_survives_additive_save_and_rebuild(client):
    model_name = await _save_build_define(client)
    wider = [*WORKING_SET, THIRD_ROW]
    save_result = await client.call_tool("save_core_schema", {"working_set": wider})
    assert _payload(save_result)["model_invalidated"] is False
    build_result = await client.call_tool("build_core_table", {})
    assert _payload(build_result)["model_invalidated"] is False
    result = await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "measures": ["avg_one"],
            "chart_spec": {"mark": "bar"},
            "label": "additive save keeps the model queryable",
        },
    )
    assert result is not None
