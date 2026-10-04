"""Pins the certified-map plumbing: the {var_name: code} projection off a
core table schema, the breaking-change comparison that decides whether a
core table rebuild may keep its core-model YAML, and registry clearing on
invalidate + reset."""

import pytest

from osaa_metrics.mcp import session_state
from osaa_metrics.mcp.session_state import is_breaking_change, schema_map

ROWS_V1 = [
    {"code": "WB-GDP", "description": "GDP growth", "var_name": "gdp_growth"},
    {
        "code": "UIS-CR1",
        "description": "Primary completion",
        "var_name": "school_completion",
    },
]


def test_schema_map_projects_pairs():
    assert schema_map(ROWS_V1) == {
        "gdp_growth": "WB-GDP",
        "school_completion": "UIS-CR1",
    }


def test_identical_map_is_not_breaking():
    assert is_breaking_change(schema_map(ROWS_V1), ROWS_V1) is False


def test_additive_change_is_not_breaking():
    wider = [
        *ROWS_V1,
        {"code": "WB-FDI", "description": "FDI", "var_name": "fdi_inflows"},
    ]
    assert is_breaking_change(schema_map(ROWS_V1), wider) is False


def test_removed_column_is_breaking():
    assert is_breaking_change(schema_map(ROWS_V1), ROWS_V1[:1]) is True


def test_repointed_var_name_is_breaking():
    # same var_name now points at a different indicator code — the
    # silent-wrong-numbers case a certified core semantic table must not survive
    swapped = [
        ROWS_V1[0],
        {
            "code": "UIS-CR2",
            "description": "Upper secondary",
            "var_name": "school_completion",
        },
    ]
    assert is_breaking_change(schema_map(ROWS_V1), swapped) is True


def test_empty_new_rows_is_breaking():
    assert is_breaking_change(schema_map(ROWS_V1), []) is True


@pytest.mark.asyncio
async def test_define_stamps_certified_map(ctx):
    await session_state.save_core_schema(ctx, ROWS_V1, "2026-08-26T00:00:00+00:00")
    await session_state.set_core_semantic_table(ctx, object(), "m: {}")
    recipe = await session_state.get_recipe(ctx)
    assert recipe.core_model_certified_map == schema_map(ROWS_V1)


@pytest.mark.asyncio
async def test_invalidate_clears_map_and_session_registry_keys(ctx):
    await session_state.save_core_schema(ctx, ROWS_V1, "2026-08-26T00:00:00+00:00")
    await session_state.set_core_semantic_table(ctx, object(), "m: {}")
    models = {f"{ctx.session_id}:m": object(), "other-session:m": object()}
    await session_state.invalidate_after_save(ctx, models=models)
    recipe = await session_state.get_recipe(ctx)
    assert recipe.core_model_certified_map is None
    assert recipe.core_model_yaml is None
    assert list(models) == ["other-session:m"]  # only this session's keys die


@pytest.mark.asyncio
async def test_reset_clears_session_registry_keys(ctx):
    # reset_state must drop this session's registry entries too, not just the
    # recipe and cache — otherwise a stale core semantic table survives a reset
    models = {f"{ctx.session_id}:m": object(), "other-session:m": object()}
    await session_state.reset(ctx, models=models)
    assert list(models) == ["other-session:m"]
