"""Core-table CSV loading and validation."""

from pathlib import Path

import polars as pl
import pytest

from osaa_metrics.core_table import load_from_csv

FIXTURE = Path(__file__).parent / "fixtures" / "core_table_sample.csv"


def _fake_cache_df() -> pl.DataFrame:
    """Gold-meta catalogue shape, covering the codes the CSV fixture carries."""
    return pl.DataFrame(
        {
            "source": ["WB"] * 3,
            "database": ["WDI"] * 3,
            "indicator_code": ["NY.GDP.MKTP.CD", "NY.GDP.MKTP.KD.ZG", "SP.POP.TOTL"],
            "meta_indicator_name": ["GDP", "GDP growth", "Population"],
            "meta_indicator_description": ["d1", "d2", "d3"],
            "cover_world_pct": [99.0, 98.0, 100.0],
            "year_start": [1960, 1961, 1960],
            "year_end": [2024, 2024, 2024],
        }
    )


def test_load_from_csv_returns_enriched_and_unknown():
    matched, unknown = load_from_csv(FIXTURE, _fake_cache_df())
    assert unknown == []
    assert len(matched) == 3
    assert set(matched["var_name"].to_list()) == {"gdp_usd", "gdp_growth", "pop_total"}
    assert "source" in matched.columns
    assert "year_start" in matched.columns


def test_load_from_csv_surfaces_unknown_codes(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("code,description,var_name\nFAKE.NOPE,Nonexistent,nope\n")
    matched, unknown = load_from_csv(csv, _fake_cache_df())
    assert unknown == ["FAKE.NOPE"]
    assert len(matched) == 0


def test_load_from_csv_rejects_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("code,description\nNY.GDP.MKTP.CD,GDP\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_from_csv(csv, _fake_cache_df())
