"""``year_from`` is interpolated into the PIVOT's SQL by f-string, and callers
reach ``build_wide_table`` directly — the marimo notebooks under ``app/`` and
``examples/`` do — with no MCP boundary validating arguments in front of it. So the substrate
coerces ``year_from`` to an int itself rather than trusting a caller-side check,
the same coercion ``top_k`` gets in ``search_indicators``."""

import ibis
import polars as pl
import pytest

from osaa_metrics.core_table import build_wide_table

CORE_TABLE = pl.DataFrame(
    {
        "var_name": ["x"],
        "indicator_code": ["C1"],
        "source": ["s"],
        "database": ["d"],
    }
)


def test_year_from_string_injection_never_reaches_sql():
    con = ibis.duckdb.connect()
    with pytest.raises((TypeError, ValueError)):
        build_wide_table(CORE_TABLE, con=con, year_from="2000 OR 1=1")  # type: ignore[arg-type]
