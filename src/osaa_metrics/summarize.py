"""Core-table SUMMARIZE → markdown rendering.

Pure substrate: no MCP imports. Importable by any caller that imports the
package, without loading FastMCP. The MCP layer's ``build_core_summary``
tool delegates here.

Public surface:
- ``build_core_summary_payload_from_handle(tbl)`` — given an ibis.Table
  pointing at the ``core`` temp table inside its connection, return the
  markdown summary (shape line + DuckDB SUMMARIZE table).
"""

from __future__ import annotations

import math
from typing import Any

import ibis

# The columns DuckDB's SUMMARIZE emits, in the order it emits them. Kept
# verbatim — no renames — so the rendered table lines up with what a reader
# gets from running SUMMARIZE themselves.
_SUMMARIZE_COLS = [
    "column_name",
    "column_type",
    "min",
    "max",
    "approx_unique",
    "avg",
    "std",
    "q25",
    "q50",
    "q75",
    "count",
    "null_percentage",
]
_INTEGER_CELLS = {"approx_unique", "count"}
_LIGHT_FMT_CELLS = {"min", "max", "avg", "std", "q25", "q50", "q75", "null_percentage"}


def _fmt_int(v: Any) -> str:
    """Comma-thousands formatting for integer-valued cells (count, approx_unique)."""
    if v is None:
        return ""
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_num(v: Any) -> str:
    """4-sig-fig 'g' for numeric stats; exponent form outside [1e-3, 1e5]; '' for None."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        # Non-numeric (e.g. lexicographic min/max on a VARCHAR column).
        return str(v)
    if math.isnan(f):
        return ""
    if f == 0:
        return "0"
    abs_f = abs(f)
    if abs_f >= 1e5 or abs_f < 1e-3:
        return f"{f:.3e}"
    return f"{f:.4g}"


def _format_summarize_row(cells: dict[str, Any]) -> list[str]:
    """Format one row of SUMMARIZE output for the markdown table."""
    out: list[str] = []
    for col in _SUMMARIZE_COLS:
        v = cells.get(col)
        if col in _INTEGER_CELLS:
            out.append(_fmt_int(v))
        elif col in _LIGHT_FMT_CELLS:
            out.append(_fmt_num(v))
        else:  # text cells, shown as-is
            out.append("" if v is None else str(v))
    return out


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Pipe-table renderer with column-aligned padding so it reads as plain text too."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)
    sep = "|".join("-" * (w + 2) for w in widths)
    header_line = (
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    )
    sep_line = "|" + sep + "|"
    body_lines = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def build_core_summary_payload_from_handle(tbl: Any) -> str:
    """Render SUMMARIZE markdown for a core_tbl ibis Table that already exists.

    Building the core table is the caller's job — ``build_core_table`` does it
    in the MCP server. The return is one markdown string, ready to render
    verbatim; nothing here shapes it for a particular transport.
    """
    # build_wide_table created a CREATE OR REPLACE TEMP TABLE core in this
    # connection; the ibis Table holds a reference so the connection is alive.
    con = ibis.get_backend(tbl)
    summarize_df = con.raw_sql("SUMMARIZE core").pl()
    n_rows = con.raw_sql("SELECT count(*) AS n FROM core").pl()["n"].item()
    n_cols = summarize_df.height  # SUMMARIZE returns one row per column

    headers = list(_SUMMARIZE_COLS)
    rows = [_format_summarize_row(r) for r in summarize_df.to_dicts()]
    table_md = _markdown_table(headers, rows)
    shape_line = f"core table — shape: {n_cols} cols × {n_rows:,} rows"
    return f"{shape_line}\n\n{table_md}"
