"""MCP tool implementations — input hardening in front of thin wrappers over
the ``osaa_metrics`` library functions the tools delegate to."""

from __future__ import annotations

import json
import re
import secrets
from datetime import UTC, datetime
from typing import Any

import polars as pl
from fastmcp.exceptions import ToolError

from osaa_metrics.core_table import build_wide_table, enrich_core_table
from osaa_metrics.discovery import search_indicators_with_fallback
from osaa_metrics.mcp import session_state
from osaa_metrics.mcp.widget_loader import load_schema_text
from osaa_metrics.semantic import ModelValidationError, validate_and_load_model
from osaa_metrics.session import (
    SCHEMA_VERSION,
    QueryChartPair,
    SchemaRow,
    Session,
    SessionValidationError,
    dump_session,
    load_session,
)
from osaa_metrics.summarize import build_core_summary_payload_from_handle

# Control-char range with \t (0x09), \n (0x0a), \r (0x0d) excluded so multi-line
# free-form text (notably YAML payloads) is not rejected. Single-line fields
# (query, indicator_code, var_name) never contain these legitimately, so the
# narrower set still catches the dangerous control chars on those inputs.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_VAR_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_INDICATOR_CODE_BAD = re.compile(r"[?#]")

# Per-field length caps — size shaping, because every tool argument arrives
# from a model and need not be sane. A search box typing 256 chars is well
# into pathological territory.
KEYWORD_QUERY_MAX = 256
# Cap on rows one discover_indicators call returns. The discovery server's
# top_k field bound reads this same constant.
TOP_K_MAX = 200
# Cap on core table schema rows per Save. The session JSON schema
# (_schemas/session.schema.json, "maxItems") carries the same ceiling; the
# two must agree.
CORE_TABLE_SCHEMA_ROWS_MAX = 100
# YAML payload cap. This is ordinary size shaping, not the anchor-expansion
# defence: _DuplicateKeyLoader refuses anchors and aliases outright, so a
# billion-laughs payload never composes in the first place. A size cap could not
# have covered that anyway — the explosion happens during parsing, after the
# bytes pass the check. 64 KiB is generous for a realistic semantic-model YAML.
CORE_MODEL_YAML_MAX = 64 * 1024
# Cap forwarded to BSL query.limit so agents can't request millions of rows.
QUERY_RESULT_LIMIT_MAX = 10_000
# Cap on the combined size of dimensions + measures in a single
# query_and_chart call. A real query has single-digit fields; 64 is well
# above any legitimate ask and protects DuckDB from a pathological projection.
QUERY_FIELD_MAX = 64
# Cap on filters list. BSL composes each filter into the WHERE clause;
# 32 is generous for a hand-authored query.
QUERY_FILTER_MAX = 32
# Cap on order_by list. ORDER BY with more than a handful of keys is
# meaningless for a single chart; 16 leaves headroom for future BSL forms.
QUERY_ORDER_MAX = 16

# Vega title text does not wrap: a line longer than the chart view runs off
# the edge of the widget. The caps bound the title to one short claim and
# each subtitle line to one line of evidence; the agent breaks longer
# subtitles into a list of lines.
CHART_TITLE_TEXT_MAX = 80
CHART_SUBTITLE_LINE_MAX = 120

# Cap on the JSON envelope of a downloaded session. Generous for a real
# session (the schema is ~3 columns × ~50 indicators + a YAML; rarely past
# 50 KiB). The payload need not validate as a session to reach `json.loads`,
# so the worst case is dict-dense — an array of many small objects (e.g.
# ``[{},{},...]``), not one long string. Measured against that shape,
# `json.loads` holds roughly 24x the source text in memory as parsed Python
# objects, so 4 MiB keeps a hostile paste near 100 MB in the worst case —
# comfortably inside a 512 MB host, with headroom over the
# schema's own 100-row maxItems ceiling at typical field sizes. Emission is
# not bounded; a session that outgrows this ceiling reloads only after
# trimming its queries by hand.
SESSION_JSON_MAX = 4 * 1024 * 1024
# Cap on reset_state.reason — recorded only on the wire reply; a sentence
# is plenty.
RESET_REASON_MAX = 256


# Per-process download stash. Entries are removed on fetch (one-shot); an
# unfetched token's bytes live until the process exits.
_DOWNLOAD_TOKENS: dict[str, dict] = {}


class ToolValidationError(ValueError):
    """Raised when an MCP tool argument fails the input-hardening checks."""


def _harden_string(
    value: str,
    field: str,
    *,
    regex: re.Pattern | None = None,
    regex_msg: str = "value does not match the required pattern",
    forbid: re.Pattern | None = None,
    forbid_msg: str = "value contains a forbidden pattern",
    max_len: int | None = None,
) -> None:
    """Defense-in-depth string check: reject control chars, optionally enforce
    length / regex / forbidden characters. Raises ToolValidationError on first
    violation. The forbid and regex branches each carry their own message
    (``forbid_msg`` / ``regex_msg``) so callers can name the actual rule that
    fired."""
    if _CTRL_RE.search(value or ""):
        raise ToolValidationError(f"{field}: contains control characters")
    if max_len is not None and len(value or "") > max_len:
        raise ToolValidationError(f"{field}: exceeds {max_len}-char cap")
    if forbid is not None and forbid.search(value or ""):
        raise ToolValidationError(f"{field} {value!r}: {forbid_msg}")
    if regex is not None and not regex.match(value or ""):
        raise ToolValidationError(f"{field} {value!r}: {regex_msg}")


_KEEP_COLS = [
    "source",
    "database",
    "indicator_code",
    "meta_indicator_name",
    "similarity_score",
    "year_start",
    "year_end",
    "cover_world_pct",
    "cover_region_africa",
]


def _project_candidate_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Project a candidate-rows DataFrame down to the columns the widget displays.

    Description (`meta_indicator_description`) is dropped here even though the
    keyword filter matches against it upstream: the widget never renders it,
    and it is the longest text field on the row, so carrying it would dominate
    the payload of every search. similarity_score is present on every row;
    the keyword path fills it with NULL.
    """
    return df.select([c for c in _KEEP_COLS if c in df.columns])


def discover_indicators(
    semantic_query: str | None = None,
    keyword_query: str | None = None,
    top_k: int = 50,
    *,
    con,
    encoder,
) -> tuple[list[dict[str, Any]], bool]:
    """Filter the indicator catalogue via a single DuckDB query — both filters
    AND-composed in the same WHERE/ORDER BY clause; no Python intersection.

    semantic_query: ``list_cosine_similarity`` ranking over the catalogue.
    keyword_query:  word-boundary regex against (code, name, description) at SQL level.
    Both set: cosine ranking restricted to keyword matches.
    Neither set: empty list — the widget stays on its empty-state hint.

    Dependencies are injected: ``con`` (process-scoped DuckDB/ibis backend) and
    ``encoder`` (whatever ``EncoderProvider`` yields). ``encoder`` is unused on
    the keyword-only path so a stub or ``None`` is acceptable there.
    Returns ``(rows, degraded)`` — ``degraded`` is True when the semantic path
    failed and the search ran as a keyword fallback. Rows are projected
    to the widget's column set; up to top_k rows returned.
    """
    if semantic_query is not None:
        _harden_string(semantic_query, "semantic_query", max_len=KEYWORD_QUERY_MAX)
    if keyword_query is not None:
        _harden_string(keyword_query, "keyword_query", max_len=KEYWORD_QUERY_MAX)
    if not isinstance(top_k, int) or top_k < 1:
        raise ToolValidationError("top_k: must be a positive integer")
    top_k = min(top_k, TOP_K_MAX)

    sem = (semantic_query or "").strip()
    kw = (keyword_query or "").strip()
    if not sem and not kw:
        return [], False

    result = search_indicators_with_fallback(
        sem, encoder if sem else None, top_k, con=con, keyword_query=kw or None
    )
    return _project_candidate_cols(result.rows).to_dicts(), result.degraded


def validate_core_table_arg(rows: list[dict]) -> list[dict]:
    """Validate a core_table argument coming from the agent. Returns the cleaned rows.

    Caps at ``CORE_TABLE_SCHEMA_ROWS_MAX`` rows; checks var_name regex and
    uniqueness; rejects bad indicator codes.
    """
    if not isinstance(rows, list):
        raise ToolValidationError("core_table: must be a list of objects")
    if len(rows) > CORE_TABLE_SCHEMA_ROWS_MAX:
        raise ToolValidationError(
            f"core_table: capped at {CORE_TABLE_SCHEMA_ROWS_MAX} rows"
        )
    seen: set[str] = set()
    cleaned: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ToolValidationError("core_table: each row must be an object")
        code = row.get("code") or row.get("indicator_code")
        var_name = row.get("var_name")
        if not code:
            raise ToolValidationError("core_table row: missing 'code'")
        if not var_name:
            raise ToolValidationError("core_table row: missing 'var_name'")
        _harden_string(
            code,
            "indicator_code",
            forbid=_INDICATOR_CODE_BAD,
            forbid_msg="contains '?' or '#'",
        )
        _harden_string(
            var_name,
            "var_name",
            regex=_VAR_NAME_RE,
            regex_msg="must match ^[a-z_][a-z0-9_]*$",
        )
        if var_name in seen:
            raise ToolValidationError(f"core_table: duplicate var_name {var_name!r}")
        seen.add(var_name)
        # Accept the widget's own field name as well, mirroring the `code`
        # fallback above. Without the symmetry a raw working-set row passes
        # validation and silently stores a blank description.
        description = row.get("description") or row.get("meta_indicator_name") or ""
        cleaned.append({"code": code, "var_name": var_name, "description": description})
    return cleaned


# =========================================================================
# Composing-layer handlers — called from OSAAMetricsServer._register_tools
# =========================================================================


async def _require_fresh_handle(ctx, retry_hint: str, not_built_hint: str):
    """Raise unless the cached core table handle exists AND was built from the
    currently committed core table schema.

    Closes the window a conditional invalidation opens: a Save that is not a
    breaking change moves the committed core table schema on without wiping
    the built handle, so a read against the stale handle would otherwise
    silently answer for a core table that no longer matches what was Saved.

    With no built handle, the error tells the agent to call ``build_core_table``
    when a core table schema is saved. When none is, it first tells the agent
    to ask the user to press Save in the discovery widget, and to call
    ``open_discovery`` if no discovery widget is open in the conversation.

    ``retry_hint`` and ``not_built_hint`` let each call site name its own
    next tool in the recovery instructions. ``not_built_hint`` follows
    ``build_core_table`` in a sentence, so it ends the sentence itself.
    """
    cache = session_state.get_cache(ctx)
    recipe = await session_state.get_recipe(ctx)
    if cache.core_tbl_handle is None:
        if recipe.committed_core_table:
            raise ToolValidationError(
                f"core table not built yet — the core table schema is saved; "
                f"call build_core_table{not_built_hint}"
            )
        raise ToolValidationError(
            f"core table not built yet — no core table schema is saved. Ask the "
            f"user to press Save in the discovery widget, then call "
            f"build_core_table{not_built_hint} If no discovery widget is open in "
            f"this conversation, call open_discovery first."
        )
    committed = session_state.schema_map(recipe.committed_core_table)
    if cache.core_tbl_built_from != committed:
        raise ToolValidationError(
            f"core table is stale — the saved core table schema changed "
            f"since it was built. Call build_core_table{retry_hint}"
        )
    return cache


async def handle_build_core_table(ctx, con, catalogue_df, models: dict) -> dict:
    """Read the core table schema from session state, run the PIVOT, store
    the live core_tbl handle in the runtime cache. Returns wire-safe metadata
    only — the ibis.Table handle itself is not JSON-serializable so it stays
    in-process.

    Pre-condition: ``save_core_schema`` has been called (recipe.committed_core_table
    populated). Post-condition: ``cache.core_tbl_handle`` populated; downstream
    tools (query_and_chart, build_core_summary) read it from the cache.

    Invalidation is conditional: a rebuild whose pairs leave every certified
    ``(var_name -> code)`` pair intact keeps the core-model YAML, the core
    semantic table, and the analysis history.
    """
    recipe = await session_state.get_recipe(ctx)
    rows = recipe.committed_core_table
    if not rows:
        raise ToolValidationError(
            "no core table schema is saved yet — ask the user to press Save in "
            "the discovery widget, then call build_core_table again. If no "
            "discovery widget is open in this conversation, call open_discovery "
            "first."
        )

    try:
        enriched = enrich_core_table(rows, cache_df=catalogue_df)
    except ValueError as e:
        raise ToolValidationError(str(e)) from e
    tbl = build_wide_table(enriched, con=con)

    await session_state.set_core_tbl_handle(ctx, tbl)
    cache = session_state.get_cache(ctx)
    cache.core_tbl_built_from = session_state.schema_map(rows)

    certified = recipe.core_model_certified_map
    had_core_semantic_table = recipe.core_model_yaml is not None
    had_queries = bool(recipe.last_analysis)
    breaking = certified is None or session_state.is_breaking_change(certified, rows)
    if breaking:
        await session_state.invalidate_after_save(ctx, models=models)

    row_count = int(tbl.count().execute())
    column_count = len(tbl.columns)
    return {
        "row_count": row_count,
        "column_count": column_count,
        "built": True,
        "model_invalidated": breaking and had_core_semantic_table,
        "queries_invalidated": breaking and had_queries,
    }


async def handle_define_semantic_model(ctx, yaml_text: str, models: dict) -> dict:
    """Validate the BSL semantic-model YAML against the materialized core
    table and register the resulting semantic table under a session-namespaced
    key (``{session_id}:{yaml_name}``) so concurrent sessions don't collide.
    One core semantic table per session: this registration replaces any core
    semantic table this session registered before. Mutates the shared
    ``models`` dict so BSL's mounted ``query_model`` sees it immediately.
    """
    _harden_string(yaml_text, "yaml_text", max_len=CORE_MODEL_YAML_MAX)

    cache = await _require_fresh_handle(
        ctx,
        retry_hint=" first, then retry define_semantic_model.",
        not_built_hint=", then retry define_semantic_model.",
    )
    core_tbl = cache.core_tbl_handle

    try:
        semantic_table, normalized_yaml, summary, model_name = validate_and_load_model(
            yaml_text, core_tbl
        )
    except ModelValidationError as e:
        await session_state.set_validation_errors(ctx, e.errors)
        raise ToolError(
            f"semantic model validation failed: {json.dumps({'errors': e.errors})}"
        ) from e

    session_key = f"{ctx.session_id}:{model_name}"
    session_state.pop_session_models(ctx.session_id, models)
    models[session_key] = semantic_table
    await session_state.set_core_semantic_table(ctx, semantic_table, normalized_yaml)
    await session_state.invalidate_after_model_load(ctx)
    return {
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_name": session_key,
        "model_summary": summary,
    }


async def handle_build_core_summary(ctx) -> dict:
    """Render the core-table summary widget (markdown via DuckDB SUMMARIZE).

    Reads the materialized core table from session state; raises if Save +
    build_core_table haven't happened yet, or if the handle predates the
    committed core table schema (staleness check).
    """
    cache = await _require_fresh_handle(
        ctx, retry_hint=", then retry.", not_built_hint="."
    )
    return {"markdown": build_core_summary_payload_from_handle(cache.core_tbl_handle)}


async def handle_reset_state(
    ctx, reason: str = "user requested", *, models: dict
) -> dict:
    """Clear the recipe, the runtime cache, and this session's registered core
    semantic tables.

    ``models`` is keyword-only and required: a caller that omitted it would
    leave this session's core semantic tables resolvable after the reset.

    ``reason`` is currently recorded only on the wire reply; if we add an
    audit log later it lands there.
    """
    _harden_string(reason, "reason", max_len=RESET_REASON_MAX)
    await session_state.reset(ctx, models=models)
    return {
        "cleared_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
    }


def handle_get_bsl_schema() -> dict:
    """Return the BSL model JSON Schema dict shipped in the wheel."""
    return json.loads(load_schema_text("bsl_model.schema.json"))


def handle_get_session_schema() -> dict:
    """Return the session JSON Schema dict shipped in the wheel."""
    return json.loads(load_schema_text("session.schema.json"))


def _validate_chart_title(title) -> None:
    """Refuse a chart title the widget cannot show whole.

    ``title`` is vega-lite's title: a string, or an object whose ``text`` and
    ``subtitle`` are each a string or a list of lines. Each line is capped
    separately; a missing title is fine.
    """
    if title is None:
        return
    if isinstance(title, str):
        text, subtitle = title, None
    elif isinstance(title, dict):
        text, subtitle = title.get("text"), title.get("subtitle")
    else:
        raise ToolValidationError("chart_spec.title: must be a string or an object")

    def _lines(value):
        return value if isinstance(value, list) else [value]

    for line in _lines(text) if text is not None else []:
        if not isinstance(line, str) or len(line) > CHART_TITLE_TEXT_MAX:
            raise ToolValidationError(
                f"chart_spec.title.text: each line must be a string of at most "
                f"{CHART_TITLE_TEXT_MAX} characters"
            )
    for line in _lines(subtitle) if subtitle is not None else []:
        if not isinstance(line, str) or len(line) > CHART_SUBTITLE_LINE_MAX:
            raise ToolValidationError(
                f"chart_spec.title.subtitle: each line must be a string of at most "
                f"{CHART_SUBTITLE_LINE_MAX} characters"
            )


async def handle_query_and_chart(
    ctx,
    models: dict,
    model_name: str,
    dimensions: list[str],
    measures: list[str],
    filters,
    order_by,
    limit: int | None,
    chart_spec: dict,
    label: str,
) -> dict:
    """Compose a BSL query against ``models[model_name]`` and render via
    BSL's ``chart()``. One SQL execution per call: ``chart()`` materializes
    internally and emits the vega-lite dict directly.

    ``chart_spec`` is required and must carry a non-empty vega-lite spec —
    either a flat dict or BSL's wrapper form ``{backend, spec, format}``.
    Backend/format keys are accepted for symmetry with BSL's ``query_model``
    but the iframe always renders altair+json.

    Returns ``{query_id, vega_lite_spec, bsl_recipe_repr, pure_sql}`` on success.
    """
    # Unwrap BSL's wrapper form; pass through flat form as-is.
    # Gate must run before all other checks so a bad spec is caught first.
    spec_inner: dict | None = None
    if isinstance(chart_spec, dict):
        spec_inner = chart_spec.get("spec") if "spec" in chart_spec else chart_spec
    if not spec_inner:
        raise ToolValidationError(
            "chart_spec is required and must carry a vega-lite spec "
            "(flat dict or {backend, spec, format})."
        )
    _validate_chart_title(spec_inner.get("title"))

    field_count = len(dimensions or []) + len(measures or [])
    if field_count > QUERY_FIELD_MAX:
        raise ToolValidationError(
            f"dimensions + measures: {field_count} exceeds {QUERY_FIELD_MAX}-field cap"
        )
    if len(filters or []) > QUERY_FILTER_MAX:
        raise ToolValidationError(
            f"filters: {len(filters)} exceeds {QUERY_FILTER_MAX}-entry cap"
        )
    for i, entry in enumerate(filters or []):
        if not isinstance(entry, dict):
            raise ToolValidationError(
                f"filters[{i}]: must be a JSON object (BSL filter spec) — "
                "string filter expressions are not accepted"
            )
    if len(order_by or []) > QUERY_ORDER_MAX:
        raise ToolValidationError(
            f"order_by: {len(order_by)} exceeds {QUERY_ORDER_MAX}-entry cap"
        )
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ToolValidationError("limit: must be a positive integer")
        limit = min(limit, QUERY_RESULT_LIMIT_MAX)
    if model_name not in models:
        raise ToolError(
            f"unknown model {model_name!r}; call list_models or "
            f"define_semantic_model first."
        )

    recipe = await session_state.get_recipe(ctx)
    cache = session_state.get_cache(ctx)
    certified = recipe.core_model_certified_map
    built_from = cache.core_tbl_built_from or {}
    # A widened core table (extra columns beyond what's certified) is safe —
    # the core semantic table simply never references the new columns. Only
    # a missing or re-pointed certified (var_name -> code) pair is stale.
    out_of_date = certified is None or not session_state.pairs_intact(
        certified, built_from
    )
    if out_of_date:
        raise ToolError(
            "core semantic table is out of date with the core table — call "
            "build_core_table and define_semantic_model again, then retry."
        )

    bsl_query = models[model_name].query(
        dimensions=dimensions,
        measures=measures,
        filters=filters or [],
        order_by=order_by,
        limit=limit,
    )

    vega_dict = bsl_query.chart(spec=spec_inner, backend="altair", format="json")
    # BSL builds the chart from mark, encoding and transform only; every
    # other top-level vega-lite key is dropped. Carry the title through so
    # the chart widget shows what the agent wrote.
    if "title" in spec_inner:
        vega_dict["title"] = spec_inner["title"]
    bsl_recipe_repr = repr(bsl_query)
    pure_sql = bsl_query.sql()

    new_id = await session_state.append_query_chart_pair(
        ctx,
        dimensions=dimensions,
        measures=measures,
        filters=filters,
        order_by=order_by,
        limit=limit,
        chart_spec=spec_inner,
        sql=pure_sql,
        label=label,
    )

    return {
        "query_id": new_id,
        "vega_lite_spec": vega_dict,
        "bsl_recipe_repr": bsl_recipe_repr,
        "pure_sql": pure_sql,
    }


async def handle_download_session(ctx) -> dict:
    """Serialize current session state into a session.json envelope per
    ``osaa_metrics.session``. Wire shape ``{filename, mime_type, content}``
    returned to the agent, which hands the content to the user.
    """
    recipe = await session_state.get_recipe(ctx)
    schema_rows = [
        SchemaRow(
            code=row.get("code") or row.get("indicator_code") or "",
            description=row.get("description") or "",
            var_name=row.get("var_name") or "",
        )
        for row in recipe.committed_core_table
    ]
    queries_raw = (recipe.last_analysis or {}).get("queries", [])
    queries = [
        QueryChartPair(
            id=q["id"],
            label=q.get("label"),
            created_at=q["created_at"],
            query_args=q.get("query_args", {}),
            chart_spec=q.get("chart_spec", {}),
            sql=q.get("sql"),
        )
        for q in queries_raw
    ]
    now = datetime.now(UTC)
    session = Session(
        schema_version=SCHEMA_VERSION,
        generated_at=now.isoformat(timespec="seconds"),
        schema=schema_rows,
        yaml=recipe.core_model_yaml,
        queries=queries,
    )
    content = dump_session(session)
    return {
        "filename": f"osaa-session-{now.strftime('%Y%m%d-%H%M%S')}.json",
        "mime_type": "application/json",
        "content": content,
    }


async def handle_prepare_download(ctx, kind: str, base_url: str) -> dict:
    """Stash a downloadable artifact under a one-shot token; return its URL.

    kind 'session' -> session.json ; 'yaml' -> the core-model YAML.
    """
    if kind not in {"session", "yaml"}:
        raise ToolValidationError("kind: must be 'session' or 'yaml'")
    if kind == "session":
        payload = await handle_download_session(ctx)  # {filename, mime_type, content}
        filename = payload["filename"]
        mime = payload["mime_type"]
        content = payload["content"]
    else:
        recipe = await session_state.get_recipe(ctx)
        if not recipe.core_model_yaml:
            raise ToolValidationError("no core-model YAML in this session yet")
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        filename = f"core-model-{ts}.yaml"
        mime = "application/x-yaml"
        content = recipe.core_model_yaml
    token = secrets.token_urlsafe(16)
    _DOWNLOAD_TOKENS[token] = {
        "filename": filename,
        "mime_type": mime,
        "bytes": content.encode("utf-8"),
    }
    return {"url": f"{base_url.rstrip('/')}/download/{token}", "filename": filename}


async def handle_load_session(
    ctx, json_text: str, con, catalogue_df, models: dict
) -> dict:
    """Validate-then-write bootstrap from a session.json blob.

    Ordering:
      1. ``osaa_metrics.session.load_session`` validates and raises on failure
         before any state mutates.
      2. Build the new runtime cache locally (PIVOT, optional YAML reload).
         The PIVOT itself replaces the shared ``core`` table before the YAML
         is validated, so a YAML failure here still lands on a session whose
         core table has moved: the ``except`` branch below drops the handle
         and this session's registry entries before re-raising, rather than
         leaving either pointed at a core table nothing certified.
      3. ``session_state.replace_atomically`` swaps cache, models registry and
         recipe; kitchen first, recipe last.

    The restore owns the whole registry slot for this session: step 3 drops
    every entry this session had registered and puts back only the model the
    restored core-model YAML defines, under a session-namespaced key
    (``{session_id}:{yaml_name}``) so concurrent sessions restoring the same
    file don't overwrite each other's entries. Models certified against the
    superseded core table must not outlive it — the restored core table can
    carry the same column names over different indicators, which no later
    guard can detect. A session file with no YAML therefore leaves this
    session with no model registered.

    If step 3's recipe write raises, the kitchen is ahead of the recipe;
    the next ``build_core_table`` rebuilds the kitchen from the (unchanged)
    recipe — self-healing.
    """
    _harden_string(json_text, "json_text", max_len=SESSION_JSON_MAX)

    try:
        session = load_session(json_text)
    except SessionValidationError as e:
        raise ToolError(
            f"session validation failed: {json.dumps({'errors': e.errors})}"
        ) from e

    schema_rows = [
        {"code": r.code, "description": r.description, "var_name": r.var_name}
        for r in session.schema
    ]

    if session.yaml and not schema_rows:
        raise ToolValidationError(
            "session yaml requires a core table schema — the file carries a "
            "core-model YAML but no schema rows to validate it against."
        )

    if session.yaml:
        _harden_string(session.yaml, "yaml", max_len=CORE_MODEL_YAML_MAX)

    # Build the new kitchen entirely in local vars before touching shared state.
    new_cache = session_state.RuntimeCache()
    if schema_rows:
        try:
            enriched = enrich_core_table(schema_rows, cache_df=catalogue_df)
        except ValueError as e:
            raise ToolError(
                "core table schema invalid: "
                + json.dumps(
                    {
                        "errors": [
                            {
                                "path": "schema",
                                "message": str(e),
                                "hint": "Remove the unknown codes from the session "
                                "file's schema before restoring, or call "
                                "open_discovery so the user can pick current "
                                "indicators.",
                            }
                        ]
                    }
                )
            ) from e
        tbl = build_wide_table(enriched, con=con)
        new_cache.core_tbl_handle = tbl
        new_cache.core_tbl_built_from = session_state.schema_map(schema_rows)

    model_name: str | None = None
    session_key: str = ""
    model_entry: tuple[str, Any] | None = None
    if session.yaml and new_cache.core_tbl_handle is not None:
        try:
            semantic_table, _normalized_yaml, _summary, model_name = (
                validate_and_load_model(session.yaml, new_cache.core_tbl_handle)
            )
        except ModelValidationError as e:
            # The core table was already replaced above, so the handle this
            # session still holds now names the restored table while its
            # provenance still describes the previous one. Drop the handle so
            # every read demands a rebuild rather than answering from a core
            # table nothing was certified against. Also drop this session's
            # registry entries: BSL's mounted query_model reads the same
            # registry with no provenance check, so a model registered
            # earlier in the session would otherwise survive and answer
            # against the replaced table.
            stale_cache = session_state.get_cache(ctx)
            stale_cache.core_tbl_handle = None
            stale_cache.core_tbl_built_from = None
            session_state.pop_session_models(ctx.session_id, models)
            raise ToolError(
                f"session yaml validation failed: {json.dumps({'errors': e.errors})}"
            ) from e
        new_cache.core_semantic_table = semantic_table
        session_key = f"{ctx.session_id}:{model_name}"
        model_entry = (session_key, semantic_table)

    new_recipe = session_state.SessionRecipe(
        committed_core_table=schema_rows,
        last_saved_at=session.generated_at,
        core_model_yaml=session.yaml,
        core_model_certified_map=(
            session_state.schema_map(schema_rows) if schema_rows else None
        ),
        last_analysis={
            "queries": [
                {
                    "id": q.id,
                    "label": q.label,
                    "created_at": q.created_at,
                    "query_args": q.query_args,
                    "chart_spec": q.chart_spec,
                    "sql": q.sql,
                }
                for q in session.queries
            ]
        }
        if session.queries
        else None,
    )
    # Validate-then-write atomic swap: kitchen first (in-process dict), then
    # the registry (this session's prior entries out, the restored one in),
    # recipe last (the one persisted mutation).
    await session_state.replace_atomically(
        ctx,
        new_cache=new_cache,
        new_recipe=new_recipe,
        models=models,
        model_entry=model_entry,
    )

    return {
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_row_count": len(schema_rows),
        "model_name": session_key,
        "queries_loaded": len(session.queries),
    }
