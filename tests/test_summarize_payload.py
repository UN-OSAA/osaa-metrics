"""Pins the shape of the core-table summary payload.

build_core_summary_payload_from_handle returns markdown that the widget
renders verbatim: a shape line followed by a markdown table with one row
per column. These tests lock that shape down so a future change to the
function cannot silently alter what the widget receives."""

import ibis
import pytest

from osaa_metrics.summarize import build_core_summary_payload_from_handle


@pytest.fixture
def core_handle():
    con = ibis.duckdb.connect()
    con.raw_sql(
        "CREATE OR REPLACE TEMP TABLE core AS "
        "SELECT * FROM (VALUES "
        "('SEN', 2.5, 11.0), ('KEN', 4.1, 9.3), ('NGA', 3.2, NULL)"
        ") AS t(iso3, gdp_growth, inflation)"
    )
    return con.table("core")


def test_payload_shape_line(core_handle):
    md = build_core_summary_payload_from_handle(core_handle)
    assert md.startswith("core table — shape: 3 cols × 3 rows")


def test_payload_has_markdown_table_with_one_row_per_column(core_handle):
    md = build_core_summary_payload_from_handle(core_handle)
    lines = md.splitlines()
    assert lines[2].startswith("|")  # header row
    assert lines[3].startswith("|")  # separator
    body = [line for line in lines[4:] if line.startswith("|")]
    assert len(body) == 3  # one row per column
    assert any("gdp_growth" in line for line in body)


def test_payload_is_plain_string(core_handle):
    md = build_core_summary_payload_from_handle(core_handle)
    assert isinstance(md, str)
