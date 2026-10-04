"""DuckDB-native semantic search over the gold meta indicator catalogue.

Cosine similarity runs inside DuckDB via ``list_cosine_similarity``, so the
embeddings stay in the database and never materialise in Python. Keyword
filtering composes into the same query via DuckDB's ``regexp_matches``, so
ranking and filtering resolve in one round trip.

Every helper here takes a caller-provided ibis backend — the substrate never
opens its own connection. SQL references the flat ``meta`` view that
``config.build_connection`` creates over the parquet source (no schema
prefix), per the substrate flatness rule.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, NamedTuple

import polars as pl

from osaa_metrics.encoders.base import Encoder

if TYPE_CHECKING:
    import ibis


logger = logging.getLogger(__name__)


_META_COLS = """
    source, database, indicator_code, meta_indicator_name,
    meta_indicator_description, meta_indicator_type,
    cover_world_pct, cover_region_africa, cover_region_americas,
    cover_region_asia, cover_region_europe, cover_region_oceania,
    year_start, year_end, stat_cagr_pct, indicator_text
"""


def load_embedding_cache(*, con: ibis.BaseBackend) -> pl.DataFrame:
    """Load the gold-meta catalogue (without the embedding column) into a
    polars DataFrame — reference data the MCP server holds for the life of
    the process. :func:`osaa_metrics.core_table.enrich_core_table` reads it to
    back-fill ``(source, database)`` from indicator codes when the core table
    is built. The embedding column is left behind: ranking happens in SQL, so
    nothing needs the vectors in Python."""
    return con.raw_sql(
        f"""
        SELECT {_META_COLS}
        FROM meta
        WHERE embedding IS NOT NULL
        """  # noqa: S608  # _META_COLS is a module constant; nothing from the caller is interpolated
    ).pl()


def search_indicators(
    query: str,
    encoder: Encoder | None,
    top_k: int,
    *,
    con: ibis.BaseBackend,
    keyword_query: str | None = None,
) -> pl.DataFrame:
    """Single DuckDB query covering every combination of semantic and
    keyword input:

    - semantic + keyword: cosine ranking, restricted to keyword matches.
    - semantic only: cosine ranking over the catalogue.
    - keyword only (``query`` empty/whitespace): keyword filter, ordered by
      ``cover_world_pct DESC`` as a sensible default; encoder is unused so
      callers may pass ``None``.
    - neither: keyword-less browse, ordered by ``cover_world_pct DESC``.

    The qvec is bound as a prepared-statement parameter so the SQL parser
    never sees user-controlled bytes. ``con`` is the caller-provided ibis
    backend — typically the session-scoped connection from
    :class:`osaa_metrics.mcp.providers.DataSourceProvider`.
    """
    semantic = bool(query and query.strip())
    keyword = bool(keyword_query and keyword_query.strip())

    if semantic:
        if encoder is None:
            raise ValueError("encoder is required when semantic query is non-empty")
        qvec = encoder.encode([query], normalize_embeddings=True)[0]
        qvec_list = [float(x) for x in qvec]
        score_expr = (
            "ROUND(list_cosine_similarity(embedding, ?::FLOAT[1024])::DOUBLE, 4)"
        )
        order = "similarity_score DESC"
        params: list = [qvec_list]
    else:
        # Keyword-only / browse mode: no embedding lookup; fixed sentinel
        # keeps the projected column shape stable across modes.
        score_expr = "CAST(NULL AS DOUBLE)"
        order = "cover_world_pct DESC NULLS LAST"
        params = []

    if keyword:
        # Word-boundary regex on lowercased text; re.escape() neutralises any
        # regex metachars in user input before we wrap. (?:^|\W) / (?:\W|$)
        # handles both alphanumeric tokens like "gdp" and non-word tokens
        # like "C++" or "R&D"; the plain \b form fails on the latter because
        # both boundaries require a word/non-word transition that doesn't
        # exist when the token itself starts or ends with non-word chars.
        kw_pattern = rf"(?:^|\W){re.escape(keyword_query.lower())}(?:\W|$)"
        where = """
            WHERE embedding IS NOT NULL
              AND (regexp_matches(lower(indicator_code), ?)
                   OR regexp_matches(lower(meta_indicator_name), ?)
                   OR regexp_matches(lower(coalesce(meta_indicator_description, '')), ?))
        """
        params.extend([kw_pattern, kw_pattern, kw_pattern])
    else:
        where = "WHERE embedding IS NOT NULL"

    sql = f"""
        SELECT {_META_COLS},
               {score_expr} AS similarity_score
        FROM meta
        {where}
        ORDER BY {order}
        LIMIT ?
    """  # noqa: S608  # _META_COLS and score_expr are module literals; user text travels in params
    params.append(int(top_k))

    # `con.raw_sql(sql, parameters=...)` delegates to DuckDBPyConnection.execute,
    # which honours parameter binding — so the qvec and keyword patterns bind
    # without reaching into the backend's private `con.con` attribute.
    # `.arrow()` returns a RecordBatchReader; `.read_all()` collects it into a
    # Table up front so a zero-row result still carries its schema — handing
    # the reader straight to `pl.from_arrow` raises on zero batches instead.
    return pl.from_arrow(con.raw_sql(sql, parameters=params).arrow().read_all())


class SearchResult(NamedTuple):
    """Outcome of a resilient discovery search.

    rows: the matched indicator rows — semantic-ranked, or the keyword
        fallback's rows.
    degraded: True when a semantic search was requested but the encoder was
        unavailable and the query ran as a keyword fallback instead. The
        server turns this flag into the discovery widget's
        "ranking unavailable" note.
    """

    rows: pl.DataFrame
    degraded: bool


def search_indicators_with_fallback(
    query: str,
    encoder: Encoder | None,
    top_k: int,
    *,
    con: ibis.BaseBackend,
    keyword_query: str | None = None,
) -> SearchResult:
    """Run :func:`search_indicators`, dropping to the keyword fallback when
    the encoder fails.

    Any exception from the semantic path lands here, whatever its shape —
    timeouts, missing weights, a broken torch install, a malformed response
    from the encoder endpoint. Each becomes a keyword search over
    ``keyword_query`` when the caller supplied one and over the semantic
    ``query`` text verbatim otherwise, with the result flagged ``degraded``.
    This is the encoder floor: a search request never surfaces an encoder
    error to the user.
    """
    try:
        rows = search_indicators(
            query, encoder, top_k, con=con, keyword_query=keyword_query
        )
        return SearchResult(rows, degraded=False)
    except Exception as exc:  # noqa: BLE001  # any encoder or search failure degrades to the keyword fallback, flagged degraded
        fallback_keyword = (
            keyword_query if (keyword_query and keyword_query.strip()) else query
        )
        logger.warning(
            "Encoder unavailable; falling back to keyword search (keyword=%r): %s",
            fallback_keyword,
            exc,
        )
        rows = search_indicators(
            "", None, top_k, con=con, keyword_query=fallback_keyword
        )
        return SearchResult(rows, degraded=True)
