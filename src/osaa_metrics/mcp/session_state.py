"""Recipe-vs-kitchen session state.

Two surfaces, two storage homes:

- **Recipe** (``SessionRecipe``): JSON-safe scalars that fully describe analyst
  intent — queries, the committed core table schema rows, the loaded YAML, the
  ``{var_name → code}`` map that YAML was certified against, timestamps, the
  in-flight analysis args. Persisted in the module-level
  ``_recipe_by_session`` dict keyed by ``ctx.session_id`` (which IS shared
  across mounted sub-servers, unlike ``ctx.set_state``, which is scoped to
  each FastMCP instance's own state store). Round-trippable through
  ``json.dumps(asdict(recipe))``.

- **Kitchen** (``RuntimeCache``): per-session Python objects rebuildable from
  the recipe — chiefly the ``core_tbl`` handle, which names a temp table on
  the running DuckDB connection. Persisted in
  ``_cache_by_session`` keyed by ``ctx.session_id``. The connection is
  process-scoped (``DataSourceProvider.get_con()``): built once on first use
  and reused thereafter; ``build_wide_table`` runs
  ``CREATE OR REPLACE TEMP TABLE core`` on it.

The kitchen is fully rebuildable from the recipe plus the data connection:
``enrich_core_table(recipe.committed_core_table, cache_df=catalogue_df)``
feeds ``build_wide_table(..., con=data_source.get_con())`` to produce the
same handle, and ``validate_and_load_model(recipe.core_model_yaml, core_tbl)``
produces the same semantic table.

``ctx.set_state(..., serializable=False)`` exists but is *request-scoped only*
(per FastMCP 3 docs) — it cannot hold the live ``ibis.Table`` across tool
calls. ``ctx.set_state(..., serializable=True)`` IS session-scoped but writes
into the per-FastMCP-instance ``_state_store`` — mounted sub-servers have
their own stores, so a tool from the mounted DiscoveryMCPServer would not see
state written by the composing OSAAMetricsServer. That's why both the kitchen
and the recipe live in our own module-level dicts keyed by
``ctx.session_id`` (which IS shared across mounts).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import Context

# Annotation-only stand-in for the BSL SemanticTable type. Widening it to
# ``Any`` keeps ``boring_semantic_layer`` out of this module's imports, which
# is what lets the BSL-free ``osaa_metrics.mcp_discovery`` import session
# state at all.
SemanticTable = (
    Any  # boring_semantic_layer.expr.SemanticTable; widened to avoid eager import
)


@dataclass
class SessionRecipe:
    """JSON-safe durable session state. Owned by ``_recipe_by_session`` keyed
    by ``ctx.session_id``."""

    semantic_query: str = ""
    keyword_query: str = ""
    committed_core_table: list[dict] = field(default_factory=list)
    last_saved_at: str | None = None
    core_model_yaml: str | None = None
    last_validation_errors: list[dict] | None = None
    last_analysis: dict | None = None
    core_model_certified_map: dict[str, str] | None = None


@dataclass
class RuntimeCache:
    """Derived, process-bound. Lives in ``_cache_by_session[ctx.session_id]``.

    Rebuildable at any time from ``SessionRecipe`` + the data connection.
    """

    core_tbl_handle: Any = None  # ibis.Table | None
    core_semantic_table: SemanticTable = None
    core_tbl_built_from: dict[str, str] | None = None


def schema_map(rows: list[dict]) -> dict[str, str]:
    """Project core table schema rows to their ``{var_name → code}`` pairs —
    the alignment identity the invalidation guard compares.

    Raises ``KeyError`` on a row carrying neither ``code`` nor
    ``indicator_code``: a ``None`` code would compare equal to an absent
    ``var_name`` and read as an intact pair, which is a false safe verdict in
    the one direction that returns wrong numbers silently."""
    return {
        r["var_name"]: (r["code"] if "code" in r else r["indicator_code"]) for r in rows
    }


def pairs_intact(certified: dict[str, str], candidate: dict[str, str]) -> bool:
    """True when every certified ``{var_name → code}`` pair still holds in
    ``candidate``. Pairs in ``candidate`` that ``certified`` never saw are
    irrelevant — a core semantic table cannot reference a column it has never
    seen. This is the single safety rule behind both the staleness guard on
    reads and the invalidation decision on writes; it lives in one place so
    the two cannot drift apart."""
    return all(candidate.get(var) == code for var, code in certified.items())


def is_breaking_change(certified: dict[str, str], new_rows: list[dict]) -> bool:
    """True when the new core table schema rows break artifacts certified
    against ``certified``: a certified ``var_name`` is gone, or now maps to a
    different indicator code. Additions never break — a core semantic table
    cannot reference a column it has never seen."""
    return not pairs_intact(certified, schema_map(new_rows))


# Kitchen state — LRU-capped at ``_CACHE_MAX_SESSIONS`` entries.
#
# Each entry is a ``RuntimeCache``, chiefly an ibis.Table handle (a small
# reference; the table itself is a DuckDB temp table living inside the shared
# connection). The cap bounds how many entries the process keeps alive at
# once; the DuckDB connection bounds its own temp-table memory.
#
# Eviction is LRU-by-access: any call to ``get_cache`` or the direct-write
# helpers (``replace_atomically``) touches the entry to most-recently-used;
# the oldest entry is popped if the cap is exceeded. An evicted session does
# NOT rebuild automatically: its next tool call gets a clean instructional
# error ("core table not built yet" / staleness guard) and the agent
# re-runs build_core_table from the recipe, or the user restores via
# load_session — which rebuilds table, model, and registry entry itself.
# The cap is safe because the recipe (the source of truth) is never
# evicted, only the cheap-to-rebuild kitchen.
#
# The recipe dict is deliberately NOT capped: it carries analyst intent that
# is expensive to lose silently, and it is the source of truth every kitchen
# rebuild reads. Bound the kitchen; keep the recipe.
_CACHE_MAX_SESSIONS = 16

_cache_by_session: OrderedDict[str, RuntimeCache] = OrderedDict()

# Recipe state lives in our own session-keyed dict instead of
# ``ctx.set_state("recipe", ...)``. FastMCP scopes ``_state_store`` per
# server instance, so the composing OSAAMetricsServer and the mounted
# DiscoveryMCPServer have separate stores — writes from one are invisible to
# the other. ``ctx.session_id``, by contrast, IS shared across mounts.
_recipe_by_session: dict[str, dict] = {}


def _touch_cache(session_id: str) -> None:
    """Mark ``session_id`` most-recently-used and evict oldest if over the cap.
    Called by every code path that reads or writes ``_cache_by_session``."""
    _cache_by_session.move_to_end(session_id)
    while len(_cache_by_session) > _CACHE_MAX_SESSIONS:
        _cache_by_session.popitem(last=False)


# --- Public API ---


async def get_state(ctx: Context) -> dict:
    """Return the recipe as a JSON-safe dict — every ``SessionRecipe`` field,
    with a session that has never written one answering as a default
    recipe rather than raising."""
    raw = _recipe_by_session.get(ctx.session_id)
    if raw is None:
        return asdict(SessionRecipe())
    return dict(raw)


async def get_recipe(ctx: Context) -> SessionRecipe:
    """Return this session's recipe as a SessionRecipe, or a default one if
    the session has none yet."""
    raw = _recipe_by_session.get(ctx.session_id)
    return SessionRecipe(**raw) if raw else SessionRecipe()


async def _save_recipe(ctx: Context, recipe: SessionRecipe) -> None:
    _recipe_by_session[ctx.session_id] = asdict(recipe)


def get_cache(ctx: Context) -> RuntimeCache:
    """Return the RuntimeCache for the current session, creating it lazily.
    Marks the entry as most-recently-used; evicts the oldest if the LRU cap
    is exceeded.

    Sync (not async) because it touches only our module-level dict, not ctx state.
    """
    if ctx.session_id not in _cache_by_session:
        _cache_by_session[ctx.session_id] = RuntimeCache()
    _touch_cache(ctx.session_id)
    return _cache_by_session[ctx.session_id]


# --- Mutators (all async, all take ctx) ---


async def set_queries(
    ctx: Context, semantic_query: str | None, keyword_query: str | None
) -> None:
    """Store the latest semantic and keyword query on the recipe; ``None``
    is stored as an empty string."""
    recipe = await get_recipe(ctx)
    recipe.semantic_query = semantic_query or ""
    recipe.keyword_query = keyword_query or ""
    await _save_recipe(ctx, recipe)


async def save_core_schema(ctx: Context, rows: list[dict], saved_at: str) -> None:
    """Persist the committed core table schema rows to the recipe.

    ``save_core_schema`` writes the recipe only; the PIVOT and the cache-handle
    stash happen in the composing layer's ``build_core_table``.
    """
    recipe = await get_recipe(ctx)
    recipe.committed_core_table = [dict(r) for r in rows]
    recipe.last_saved_at = saved_at
    await _save_recipe(ctx, recipe)


async def set_core_tbl_handle(ctx: Context, tbl_handle: Any) -> None:
    """Stash a freshly built core_tbl handle into the kitchen.

    Companion to ``save_core_schema``: schema → recipe, handle → kitchen.
    """
    cache = get_cache(ctx)
    cache.core_tbl_handle = tbl_handle


async def set_core_semantic_table(
    ctx: Context, semantic_table: Any, yaml_text: str
) -> None:
    """Record a successfully validated core semantic table, stamping the
    ``{var_name → code}`` map it was certified against. Clears prior
    validation errors so the agent doesn't see stale failures."""
    recipe = await get_recipe(ctx)
    recipe.core_model_yaml = yaml_text
    recipe.core_model_certified_map = schema_map(recipe.committed_core_table)
    recipe.last_validation_errors = None
    await _save_recipe(ctx, recipe)

    cache = get_cache(ctx)
    cache.core_semantic_table = semantic_table


async def set_validation_errors(ctx: Context, errors: list[dict] | None) -> None:
    """Store normalized validation errors without touching the core semantic
    table. Used by define_semantic_model on failure so the agent can fix the
    YAML and retry."""
    recipe = await get_recipe(ctx)
    recipe.last_validation_errors = errors
    await _save_recipe(ctx, recipe)


_QUERY_ID_DIGITS = 3


def _next_query_id(queries: list[dict]) -> str:
    next_num = 1 + max(
        (
            int(q["id"].split("-", 1)[1])
            for q in queries
            if q.get("id", "").startswith("q-")
        ),
        default=0,
    )
    return f"q-{next_num:0{_QUERY_ID_DIGITS}d}"


async def append_query_chart_pair(
    ctx: Context,
    *,
    dimensions: list[str],
    measures: list[str],
    filters,
    order_by,
    limit,
    chart_spec: dict,
    sql: str,
    label: str,
) -> str:
    """Append a query/chart pair to ``recipe.last_analysis["queries"]`` with a
    server-assigned monotonic ``q-NNN`` id. Returns the new id. A later
    ``invalidate_after_save`` clears the list along with ``core_model_yaml``
    and the certified map; ``invalidate_after_model_load`` clears the list
    alone."""
    recipe = await get_recipe(ctx)
    queries = (
        list(recipe.last_analysis.get("queries", [])) if recipe.last_analysis else []
    )
    new_id = _next_query_id(queries)
    queries.append(
        {
            "id": new_id,
            "label": label,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "query_args": {
                "dimensions": dimensions,
                "measures": measures,
                "filters": filters,
                "order_by": order_by,
                "limit": limit,
            },
            "chart_spec": chart_spec,
            "sql": sql,
        }
    )
    recipe.last_analysis = {"queries": queries}
    await _save_recipe(ctx, recipe)
    return new_id


async def replace_atomically(
    ctx: Context,
    *,
    new_cache: RuntimeCache,
    new_recipe: SessionRecipe,
    models: dict | None = None,
    model_entry: tuple[str, Any] | None = None,
) -> None:
    """Validate-then-write atomic swap used by ``load_session``: every slot the
    session owns changes here and nowhere else in a restore.

    The new cache is assigned first (in-process dict set; atomic under the
    GIL). Then, when the models registry is provided, this session's existing
    registry entries are dropped and ``model_entry`` — the ``(key,
    core_semantic_table)`` pair the restored core-model YAML produced, absent
    when the session file carries no YAML — is registered in their place.
    Dropping and registering happen together, in that order, because a restore
    that registered first would delete the entry it had just added, and a
    restore that never dropped would leave models certified against the
    previous core table answering queries against the new one.

    The recipe is persisted last. If that write raises, the next
    ``build_core_table`` rebuilds the kitchen from the (unchanged) recipe —
    self-healing."""
    _cache_by_session[ctx.session_id] = new_cache
    _touch_cache(ctx.session_id)
    if models is not None:
        pop_session_models(ctx.session_id, models)
        if model_entry is not None:
            key, semantic_table = model_entry
            models[key] = semantic_table
    await _save_recipe(ctx, new_recipe)


async def invalidate_after_save(ctx: Context, *, models: dict) -> None:
    """A BREAKING core table change invalidates every downstream slot — the
    loaded core semantic table, its certified map, the accumulated
    query/chart pairs, and this session's registry entries. Callers decide
    *whether* the change is breaking via ``is_breaking_change``; this
    function only ever fires scorched-earth.

    ``models`` is keyword-only and required: BSL's mounted ``query_model``
    reads the same registry with no provenance check, so a caller that
    skipped this pop would leave a discarded core semantic table resolvable
    and answering queries against a core table it was never certified
    against."""
    recipe = await get_recipe(ctx)
    recipe.core_model_yaml = None
    recipe.core_model_certified_map = None
    recipe.last_analysis = None
    await _save_recipe(ctx, recipe)

    cache = get_cache(ctx)
    cache.core_semantic_table = None

    pop_session_models(ctx.session_id, models)


def pop_session_models(session_id: str, models: dict) -> None:
    """Drop every registry entry belonging to ``session_id`` (keys are
    ``{session_id}:{model_name}``)."""
    prefix = f"{session_id}:"
    for key in [k for k in models if k.startswith(prefix)]:
        del models[key]


async def invalidate_after_model_load(ctx: Context) -> None:
    """A new core semantic table invalidates the prior query/chart but not
    the core table itself."""
    recipe = await get_recipe(ctx)
    recipe.last_analysis = None
    await _save_recipe(ctx, recipe)


async def reset(ctx: Context, *, models: dict) -> None:
    """Wipe this session's slot. Used by ``reset_state``.

    Drops the cache entry, the recipe, and this session's registry entries,
    so the next call starts from nothing — including BSL's mounted query
    surface.

    ``models`` is keyword-only and required: BSL's mounted ``query_model``
    reads the same registry with no provenance check, so a caller that
    skipped this pop would leave this session's core semantic tables
    resolvable after the reset.
    """
    _recipe_by_session.pop(ctx.session_id, None)
    _cache_by_session.pop(ctx.session_id, None)
    pop_session_models(ctx.session_id, models)
