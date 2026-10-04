"""Core-table registry + pivot.

Two responsibilities live here because both belong to the same surface:

- **Registry IO + slugging**: load 3-column ``(code, description, var_name)`` CSVs;
  derive ``var_name`` slugs from indicator names; dedup collisions. That
  3-column registry is the **core table schema**.

- **Pivot build**: take the saved registry and ``PIVOT master`` into the wide
  ``core table`` (one column per ``var_name``).

Both consume the same registry dataframe; keeping them in one module mirrors
how callers actually use them.
"""

from __future__ import annotations

import re
from pathlib import Path

import ibis
import polars as pl

_CSV_REQUIRED = ("code", "description", "var_name")

_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "and",
    "or",
    "to",
    "for",
    "with",
    "by",
    "as",
    "is",
    "are",
    "be",
    "from",
    "that",
    "this",
    "per",
    "all",
    "total",
    "other",
    "not",
}


def load_from_csv(
    csv_path: str | Path,
    cache_df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Read a core table schema CSV (code, description, var_name); return (enriched df, list of codes not in gold meta).

    ``cache_df`` is the gold-meta catalogue (typically from
    :func:`osaa_metrics.discovery.load_embedding_cache`).
    """
    df = pl.read_csv(csv_path)
    missing = [c for c in _CSV_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. Required: {list(_CSV_REQUIRED)}"
        )
    meta = cache_df.select(
        [
            "source",
            "database",
            "indicator_code",
            "meta_indicator_name",
            "meta_indicator_description",
            "cover_world_pct",
            "year_start",
            "year_end",
        ]
    )
    enriched = df.rename({"code": "indicator_code"}).join(
        meta,
        on="indicator_code",
        how="left",
    )
    unknown = enriched.filter(pl.col("source").is_null())["indicator_code"].to_list()
    matched = enriched.filter(pl.col("source").is_not_null()).select(
        [
            "var_name",
            "indicator_code",
            "source",
            "database",
            "meta_indicator_name",
            "meta_indicator_description",
            "cover_world_pct",
            "year_start",
            "year_end",
        ]
    )
    return matched, unknown


def slugify_indicator_name(name: str) -> str:
    """Slugify an indicator name to a snake_case var_name (max 6 content words).

    The discovery widget's JS ``slugify()`` implements the same rules — same
    stopword set, same 6-word cut — so the var_name a user sees in the working
    set is the one the substrate derives. Keep the two in sync.
    """
    s = name.lower().replace("%", " pct ").replace("$", " usd ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    words = [w for w in s.split() if w and w not in _STOPWORDS]
    return "_".join(words[:6]) or "indicator"


def assign_var_names(names: list[str]) -> list[str]:
    """Slugify a list of indicator names, deduping collisions with _2, _3 suffixes.

    Tracks emitted names (not just base counts) so a literal ``gdp_2`` slugified
    before a second ``gdp`` doesn't collide with the auto-bumped ``gdp_2``.
    """
    out: list[str] = []
    used: set[str] = set()
    counts: dict[str, int] = {}
    for n in names:
        base = slugify_indicator_name(n)
        candidate = base
        while candidate in used:
            counts[base] = counts.get(base, 1) + 1
            candidate = f"{base}_{counts[base]}"
        used.add(candidate)
        out.append(candidate)
    return out


def enrich_core_table(rows: list[dict], *, cache_df: pl.DataFrame) -> pl.DataFrame:
    """Augment validated core-table registry rows with source/database from the
    gold-meta cache.

    The PIVOT in :func:`build_wide_table` needs the 4-column join key
    ``(source, database, indicator_code, var_name)``, but the agent only
    passes ``(code, var_name, description)``. This helper looks up
    ``source``/``database`` per code from ``cache_df`` (typically
    ``discovery.load_embedding_cache()``).

    Raises ValueError if any code is missing from ``cache_df``.
    """
    code_to_meta = {r["indicator_code"]: r for r in cache_df.to_dicts()}
    enriched: list[dict] = []
    unknown: list[str] = []
    for r in rows:
        meta = code_to_meta.get(r["code"])
        if meta is None:
            unknown.append(r["code"])
            continue
        enriched.append(
            {
                "source": meta["source"],
                "database": meta["database"],
                "indicator_code": r["code"],
                "var_name": r["var_name"],
            }
        )
    if unknown:
        raise ValueError(f"unknown indicator codes: {unknown}")
    return pl.DataFrame(enriched)


def _sql_quote(v: str) -> str:
    """Quote a string for inline SQL (escapes single quotes)."""
    return "'" + v.replace("'", "''") + "'"


def build_wide_table(
    core_table: pl.DataFrame,
    *,
    con: ibis.BaseBackend,
    year_from: int = 2000,
) -> ibis.Table:
    """Pivot the ``master`` long table into a wide ``core`` temp table with one
    row per country and year, from ``year_from`` on, that has a value for at
    least one indicator.

    Every ``var_name`` in ``core_table`` becomes a column, listed in the
    pivot's ``IN`` clause. An indicator with no rows from ``year_from`` on
    still gets its column, holding only NULLs.

    Runs ``CREATE OR REPLACE TEMP TABLE core`` on the caller-provided
    connection and returns the ibis handle. ``master`` is referenced flat,
    with no schema prefix, per the substrate flatness rule — it is the view
    ``config.build_connection`` creates over the parquet source.
    """
    year_from = int(year_from)
    rows_sql = ",\n        ".join(
        "({s}, {d}, {c}, {v})".format(
            s=_sql_quote(r["source"]),
            d=_sql_quote(r["database"]),
            c=_sql_quote(r["indicator_code"]),
            v=_sql_quote(r["var_name"]),
        )
        for r in core_table.iter_rows(named=True)
    )
    pivot_columns_sql = ", ".join(
        _sql_quote(v) for v in core_table["var_name"].to_list()
    )
    sql = f"""
        CREATE OR REPLACE TEMP TABLE core AS
        PIVOT (
          SELECT
            v.var_name,
            -- year is VARCHAR on purpose: a numeric year would get a
            -- quantitative chart axis that prints 2,010 instead of 2010.
            CAST(m.year AS VARCHAR) AS year,
            m.value,
            m.iso3, m.iso2, m.m49, m.country,
            m.region, m.subregion, m.intermediate_region,
            m.is_ldc, m.is_lldc, m.is_sids, m.income_group
          FROM master m
          JOIN (VALUES
        {rows_sql}
          ) AS v(source, database, indicator_code, var_name)
            ON m.source = v.source
           AND m.database = v.database
           AND m.indicator_code = v.indicator_code
          WHERE m.year >= {year_from}
        )
        ON var_name IN ({pivot_columns_sql})
        USING FIRST(value)
    """  # noqa: S608  # every VALUES row and IN value passes _sql_quote and year_from is coerced to int above
    con.raw_sql(sql)
    return con.table("core")
