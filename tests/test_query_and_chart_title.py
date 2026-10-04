"""query_and_chart carries the agent's chart title through to the rendered
vega-lite spec, and refuses titles too long to fit the chart widget."""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

GDP = "NY.GDP.MKTP.CD"

MODEL_YAML = """
econ:
  table: core_tbl
  dimensions:
    iso3: _.iso3
  measures:
    total_gdp: _.gdp.sum()
"""


async def _ready_model(client) -> str:
    rows = [{"code": GDP, "var_name": "gdp", "description": "GDP"}]
    await client.call_tool("save_core_schema", {"working_set": rows})
    await client.call_tool("build_core_table", {})
    res = await client.call_tool("define_semantic_model", {"yaml_text": MODEL_YAML})
    return res.structured_content["model_name"]


async def _chart(client, model_name: str, chart_spec: dict):
    return await client.call_tool(
        "query_and_chart",
        {
            "model_name": model_name,
            "label": "GDP by country",
            "dimensions": ["iso3"],
            "measures": ["total_gdp"],
            "limit": 5,
            "chart_spec": chart_spec,
        },
    )


@pytest.mark.asyncio
async def test_title_object_is_carried_into_vega_lite_spec(client) -> None:
    model_name = await _ready_model(client)
    title = {
        "text": "GDP is concentrated in a handful of economies",
        "subtitle": [
            "GDP (current US$), total by country",
            "Source: World Bank WDI, NY.GDP.MKTP.CD",
        ],
    }
    result = await _chart(client, model_name, {"mark": "bar", "title": title})
    assert result.structured_content["vega_lite_spec"]["title"] == title


@pytest.mark.asyncio
async def test_title_text_over_cap_is_refused(client) -> None:
    model_name = await _ready_model(client)
    long_text = "x" * 81
    with pytest.raises(ToolError) as exc_info:
        await _chart(client, model_name, {"mark": "bar", "title": {"text": long_text}})
    assert "title" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subtitle_line_over_cap_is_refused(client) -> None:
    model_name = await _ready_model(client)
    with pytest.raises(ToolError) as exc_info:
        await _chart(
            client,
            model_name,
            {"mark": "bar", "title": {"text": "Short claim", "subtitle": ["y" * 121]}},
        )
    assert "subtitle" in str(exc_info.value)
