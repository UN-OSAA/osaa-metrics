"""Pivot: long master OBT → wide core_tbl. Uses flat table names
(``master``) and a caller-provided connection."""

import ibis
import polars as pl
import pytest

from osaa_metrics.core_table import build_wide_table


@pytest.fixture
def tiny_master_con(tmp_path):
    """In-memory DuckDB with a flat ``master`` table for pivot testing.

    Mirrors what the OSAA_DATA_* connection produces: a single ``master``
    view at the root namespace, no ``silver.*`` schema prefix (substrate
    flatness).
    """
    con = ibis.duckdb.connect(":memory:")
    con.raw_sql("""
        CREATE OR REPLACE TABLE master AS
        SELECT * FROM (VALUES
          ('WB', 'WDI', 'NY.GDP.MKTP.CD', 'KEN', 'KE', 404, 'Kenya',
           'Africa', 'Eastern Africa', 'Sub-Saharan Africa',
           false, true, false, 'Lower middle income', 2020, 100.0),
          ('WB', 'WDI', 'NY.GDP.MKTP.CD', 'KEN', 'KE', 404, 'Kenya',
           'Africa', 'Eastern Africa', 'Sub-Saharan Africa',
           false, true, false, 'Lower middle income', 2021, 110.0),
          ('WB', 'WDI', 'SP.POP.TOTL', 'KEN', 'KE', 404, 'Kenya',
           'Africa', 'Eastern Africa', 'Sub-Saharan Africa',
           false, true, false, 'Lower middle income', 2020, 53.0),
          ('WB', 'WDI', 'SP.POP.TOTL', 'KEN', 'KE', 404, 'Kenya',
           'Africa', 'Eastern Africa', 'Sub-Saharan Africa',
           false, true, false, 'Lower middle income', 2021, 54.0)
        ) AS t(source, database, indicator_code, iso3, iso2, m49, country,
               region, subregion, intermediate_region,
               is_ldc, is_lldc, is_sids, income_group, year, value)
    """)
    return con


def test_build_wide_table_pivots_indicators_to_columns(tiny_master_con):
    committed = pl.DataFrame(
        {
            "source": ["WB", "WB"],
            "database": ["WDI", "WDI"],
            "indicator_code": ["NY.GDP.MKTP.CD", "SP.POP.TOTL"],
            "var_name": ["gdp_usd", "pop_total"],
        }
    )
    tbl = build_wide_table(committed, con=tiny_master_con, year_from=2020)
    df = tbl.execute()
    assert "gdp_usd" in df.columns
    assert "pop_total" in df.columns
    assert "year" in df.columns
    assert "country" in df.columns
    assert len(df) == 2  # 2 years × 1 country


def test_build_wide_table_filters_by_year_from(tiny_master_con):
    committed = pl.DataFrame(
        {
            "source": ["WB"],
            "database": ["WDI"],
            "indicator_code": ["NY.GDP.MKTP.CD"],
            "var_name": ["gdp_usd"],
        }
    )
    tbl = build_wide_table(committed, con=tiny_master_con, year_from=2021)
    df = tbl.execute()
    assert len(df) == 1
    assert df["year"][0] == "2021"


def _add_old_survey_row(con) -> None:
    """Insert an indicator whose only row, for Kenya, is from 1995."""
    con.raw_sql("""
        INSERT INTO master VALUES
          ('WB', 'WDI', 'OLD.SURVEY', 'KEN', 'KE', 404, 'Kenya',
           'Africa', 'Eastern Africa', 'Sub-Saharan Africa',
           false, true, false, 'Lower middle income', 1995, 7.0)
    """)


def test_indicator_with_no_rows_since_year_from_gets_an_empty_column(tiny_master_con):
    _add_old_survey_row(tiny_master_con)
    committed = pl.DataFrame(
        {
            "source": ["WB", "WB"],
            "database": ["WDI", "WDI"],
            "indicator_code": ["NY.GDP.MKTP.CD", "OLD.SURVEY"],
            "var_name": ["gdp_usd", "old_survey"],
        }
    )
    df = build_wide_table(committed, con=tiny_master_con, year_from=2020).to_polars()
    assert "old_survey" in df.columns
    assert df["old_survey"].null_count() == df.height
    assert df.schema["old_survey"] == df.schema["gdp_usd"]
    assert df.height == 2  # 2 years x 1 country, unchanged by the empty indicator


def test_no_indicator_with_rows_since_year_from_still_builds_every_column(
    tiny_master_con,
):
    _add_old_survey_row(tiny_master_con)
    committed = pl.DataFrame(
        {
            "source": ["WB"],
            "database": ["WDI"],
            "indicator_code": ["OLD.SURVEY"],
            "var_name": ["old_survey"],
        }
    )
    df = build_wide_table(committed, con=tiny_master_con, year_from=2020).to_polars()
    assert "old_survey" in df.columns
    assert df.height == 0


def test_var_name_with_a_quote_is_quoted_and_becomes_the_column_name(tiny_master_con):
    committed = pl.DataFrame(
        {
            "source": ["WB"],
            "database": ["WDI"],
            "indicator_code": ["NY.GDP.MKTP.CD"],
            "var_name": ["o'brien"],
        }
    )
    df = build_wide_table(committed, con=tiny_master_con, year_from=2020).to_polars()
    assert "o'brien" in df.columns
    assert df["o'brien"].null_count() == 0
