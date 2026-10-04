"""Handler bodies for the DiscoveryMCPServer tools.

CONTRACT: zero ``from boring_semantic_layer`` imports anywhere in this
sub-package.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from osaa_metrics.mcp import session_state
from osaa_metrics.mcp.tools import (
    KEYWORD_QUERY_MAX,
    _harden_string,
    validate_core_table_arg,
)

if TYPE_CHECKING:
    from fastmcp import Context


async def handle_open_discovery(
    ctx: Context,
    initial_semantic_query: str | None,
    initial_keyword_query: str | None,
) -> dict:
    """Mount the discovery widget. Persists optional seed queries into session
    state so the widget rehydrates them. Returns the current discovery state
    (the widget reads it via the App SDK)."""
    if initial_semantic_query is not None:
        _harden_string(
            initial_semantic_query,
            "initial_semantic_query",
            max_len=KEYWORD_QUERY_MAX,
        )
    if initial_keyword_query is not None:
        _harden_string(
            initial_keyword_query,
            "initial_keyword_query",
            max_len=KEYWORD_QUERY_MAX,
        )

    if initial_semantic_query is not None or initial_keyword_query is not None:
        current = await session_state.get_state(ctx)
        sem = (
            initial_semantic_query
            if initial_semantic_query is not None
            else current["semantic_query"]
        )
        kw = (
            initial_keyword_query
            if initial_keyword_query is not None
            else current["keyword_query"]
        )
        await session_state.set_queries(ctx, sem, kw)
    return await session_state.get_state(ctx)


async def handle_save_core_schema(
    ctx: Context, working_set: list[dict], *, models: dict
) -> dict:
    """Commit the working set as the core table schema. Validates the rows;
    persists to session state. Does NOT run the PIVOT — that is the composing
    layer's ``build_core_table`` tool.

    Invalidation is conditional: a Save whose pairs leave every certified
    ``(var_name -> code)`` pair intact keeps the core-model YAML, the core
    semantic table, and the analysis history. A breaking Save also drops this
    session's entries from ``models``, so the discarded core semantic table
    stops resolving the moment it is discarded rather than at the next build.
    ``models`` is keyword-only and required: a caller that omitted it would
    leave the discarded core semantic table answering queries.

    Response shape:
    ``{saved_at, row_count, schema, model_invalidated, queries_invalidated}``.
    ``saved_at`` is the ISO timestamp the widget's Save handler guards on.
    The two booleans report what the wipe actually destroyed: a core semantic
    table, saved queries, either, or neither. A session restored from a file
    that carried queries but no core-model YAML loses queries alone.
    """
    cleaned = validate_core_table_arg(working_set)
    saved_at = datetime.now(UTC).isoformat(timespec="seconds")

    recipe = await session_state.get_recipe(ctx)
    certified = recipe.core_model_certified_map
    had_core_semantic_table = recipe.core_model_yaml is not None
    had_queries = bool(recipe.last_analysis)
    breaking = certified is None or session_state.is_breaking_change(certified, cleaned)

    await session_state.save_core_schema(ctx, cleaned, saved_at)
    if breaking:
        await session_state.invalidate_after_save(ctx, models=models)
    return {
        "saved_at": saved_at,
        "row_count": len(cleaned),
        "schema": cleaned,
        "model_invalidated": breaking and had_core_semantic_table,
        "queries_invalidated": breaking and had_queries,
    }
