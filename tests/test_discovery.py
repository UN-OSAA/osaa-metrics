"""Discovery: slugify, var_name dedup, and DuckDB-native search_indicators.

Live tests route through the local ``data/`` mirror by default (conftest
sets OSAA_DATA_MASTER_URL / OSAA_DATA_META_URL when present); export those
vars to run against another source.
"""

from __future__ import annotations

import os
import re

import numpy as np
import polars as pl
import pytest

from osaa_metrics import discovery
from osaa_metrics.core_table import assign_var_names, slugify_indicator_name

needs_net = pytest.mark.skipif(
    os.environ.get("OSAA_SKIP_NET") == "1",
    reason="network-backed test skipped via OSAA_SKIP_NET=1",
)


# --- slugify / var_name ---


def test_slugify_strips_stopwords_and_punctuation():
    # `$` expands to " usd " before alphanumeric filtering, so "US$" → "us usd";
    # `per` is a stopword and gets dropped.
    assert (
        slugify_indicator_name("GDP per capita (current US$)")
        == "gdp_capita_current_us_usd"
    )


def test_slugify_drops_pct_token_idiomatically():
    # `%` expands to " pct ", so "annual %" → "annual pct".
    assert slugify_indicator_name("Trade (% of GDP)") == "trade_pct_gdp"


def test_slugify_caps_at_six_words():
    name = "Annual percentage change of GDP per capita real terms historical"
    out = slugify_indicator_name(name)
    assert out.count("_") <= 5  # 6 words → 5 underscores


def test_slugify_handles_empty():
    assert slugify_indicator_name("of the") == "indicator"


def test_assign_var_names_dedups_collisions():
    out = assign_var_names(["GDP", "Total GDP", "gdp"])
    assert len(out) == 3
    assert len(set(out)) == 3  # all unique
    assert out[0] == "gdp"
    assert "_" in out[2]  # dedup suffix


# --- load_embedding_cache shape ---


@needs_net
def test_load_embedding_cache_returns_dataframe_no_matrix(con):
    """``load_embedding_cache`` returns catalogue metadata only — no embedding
    column, no similarity matrix. Its callers need the code→(source, database)
    lookup; cosine ranking stays inside DuckDB."""
    out = discovery.load_embedding_cache(con=con)
    assert isinstance(out, pl.DataFrame)
    assert "indicator_code" in out.columns
    assert "source" in out.columns
    assert "embedding" not in out.columns
    assert "trend_over_years" not in out.columns


# --- search_indicators (DuckDB-native SQL) ---


class _FakeEncoder:
    """Deterministic encoder for tests: returns a unit vector built from a
    hash of the query string. Avoids downloading the real BGE-M3 weights."""

    def encode(self, sentences, *, normalize_embeddings: bool = True):
        rng = np.random.default_rng(abs(hash(sentences[0])) % (2**32))
        vec = rng.standard_normal(1024).astype(np.float32)
        if normalize_embeddings:
            vec = vec / np.linalg.norm(vec)
        return np.array([vec])


@pytest.fixture(scope="module")
def fake_encoder():
    return _FakeEncoder()


@pytest.fixture(scope="module")
def con():
    """Settings-driven DuckDB connection for substrate-level discovery tests.
    Routes through the local ``data/`` mirror by default (conftest sets
    OSAA_DATA_MASTER_URL / OSAA_DATA_META_URL when present)."""
    from osaa_metrics.config import build_connection, load_settings

    try:
        return build_connection(load_settings())
    except Exception as e:  # noqa: BLE001  # any failure to reach the data source skips the test rather than failing it
        pytest.skip(f"data source unreachable: {e}")


@needs_net
def test_search_indicators_top_k_respected(con, fake_encoder):
    out = discovery.search_indicators("anything", fake_encoder, top_k=3, con=con)
    assert len(out) == 3


@needs_net
def test_search_indicators_sorted_by_similarity_desc(con, fake_encoder):
    out = discovery.search_indicators("anything", fake_encoder, top_k=20, con=con)
    scores = out["similarity_score"].to_list()
    assert scores == sorted(scores, reverse=True)


@needs_net
def test_search_indicators_keyword_only_word_boundary(con, fake_encoder):
    """A keyword matches as a word, not as a substring: 'LCU' must not pull in
    'calCULated' or 'incLUded'. The match runs against (code, name,
    description) in SQL."""
    out = discovery.search_indicators(
        "anything", fake_encoder, top_k=50, con=con, keyword_query="LCU"
    )
    pattern = re.compile(r"\blcu\b", re.IGNORECASE)
    for row in out.to_dicts():
        haystack = " ".join(
            [
                row.get("indicator_code") or "",
                row.get("meta_indicator_name") or "",
                row.get("meta_indicator_description") or "",
            ]
        )
        assert pattern.search(haystack), (
            f"row {row['indicator_code']} matched 'LCU' as a substring, not a word"
        )


# --- keyword regex shape (no network needed; locks the pattern construction
#     that search_indicators feeds to DuckDB's RE2 engine) ---


def _kw_pattern(keyword: str) -> str:
    """Mirror of the pattern ``search_indicators`` builds — keeps these tests
    locked to the SQL-side regex without spinning up DuckDB."""
    return rf"(?:^|\W){re.escape(keyword.lower())}(?:\W|$)"


@pytest.mark.parametrize(
    "keyword,haystack",
    [
        ("C++", "c++ programming reference"),  # non-word token at start
        ("C++", "a guide to c++ programming"),  # non-word token mid-sentence
        ("R&D", "r&d spending as % of gdp"),  # ampersand inside token
        ("3.5%", "between 3.5% and 4.0% annual"),  # punctuation + percent
        ("gdp", "gdp per capita"),  # plain word still works
        ("gdp", "real gdp growth"),  # word mid-string
    ],
)
def test_keyword_pattern_matches_non_word_tokens(keyword, haystack):
    """Tokens carrying non-word characters (C++, R&D, 3.5%) must match. A
    ``\\b...\\b`` form cannot do it: ``\\b`` requires a word/non-word transition,
    which does not exist when the token itself starts or ends with a non-word
    character."""
    assert re.search(_kw_pattern(keyword), haystack.lower()), (
        f"pattern for {keyword!r} should match {haystack!r}"
    )


@pytest.mark.parametrize(
    "keyword,haystack",
    [
        ("gdp", "gdpr compliance audit"),  # substring guard: gdp ≠ gdpr
        ("LCU", "calculated values"),  # substring guard from prior bug
        ("LCU", "included indicators"),
    ],
)
def test_keyword_pattern_rejects_substring_matches(keyword, haystack):
    """The pattern must not match `gdp` inside `gdpr` or `LCU` inside
    `calCULated`: for alphanumeric tokens ``(?:^|\\W)...(?:\\W|$)`` still
    requires the character before and after to be non-word."""
    assert not re.search(_kw_pattern(keyword), haystack.lower()), (
        f"pattern for {keyword!r} should NOT match {haystack!r}"
    )


@needs_net
def test_search_indicators_semantic_plus_keyword_intersection(con, fake_encoder):
    """Both filters compose in one SQL query — output rows match the keyword
    pattern AND are ordered by similarity desc."""
    out = discovery.search_indicators(
        "trade balance", fake_encoder, top_k=20, con=con, keyword_query="gdp"
    )
    pattern = re.compile(r"\bgdp\b", re.IGNORECASE)
    for row in out.to_dicts():
        haystack = " ".join(
            [
                row.get("indicator_code") or "",
                row.get("meta_indicator_name") or "",
                row.get("meta_indicator_description") or "",
            ]
        )
        assert pattern.search(haystack)
    scores = out["similarity_score"].to_list()
    assert scores == sorted(scores, reverse=True)


@needs_net
def test_search_indicators_returns_expected_columns(con, fake_encoder):
    """The query returns the catalogue columns the MCP projects its candidate
    rows from, plus the description the keyword filter matches against."""
    out = discovery.search_indicators("anything", fake_encoder, top_k=1, con=con)
    expected = {
        "source",
        "database",
        "indicator_code",
        "meta_indicator_name",
        "meta_indicator_description",
        "similarity_score",
        "year_start",
        "year_end",
        "cover_world_pct",
        "cover_region_africa",
    }
    assert expected.issubset(set(out.columns))


@needs_net
def test_search_indicators_keyword_only_returns_empty_frame_on_no_match(con):
    """A keyword that matches nothing must return an empty DataFrame, not raise.

    DuckDB's ``.arrow()`` yields a RecordBatchReader; with zero rows it has no
    batches, and handing that straight to ``pl.from_arrow`` raises ('Must pass
    schema, or at least one RecordBatch') because polars chains the reader's
    batches rather than materializing it first.
    """
    out = discovery.search_indicators(
        "", None, top_k=50, con=con, keyword_query="zzz_no_such_indicator_zzz"
    )
    assert len(out) == 0


# --- search_indicators_with_fallback (resilient discovery) ---


class _BrokenEncoder:
    """Encoder whose ``encode()`` raises — one that was built but fails at
    query time. The keyword fallback catches any encoder failure."""

    def encode(self, sentences, *, normalize_embeddings: bool = True):
        raise RuntimeError("encoder failed at query time")


@needs_net
def test_with_fallback_degrades_to_keyword_on_encoder_error(con):
    result = discovery.search_indicators_with_fallback(
        "gdp", _BrokenEncoder(), top_k=5, con=con
    )
    assert result.degraded is True
    assert len(result.rows) > 0  # 'gdp' matches real indicators
    # keyword mode → similarity_score is NULL for every row
    assert result.rows["similarity_score"].null_count() == len(result.rows)


@needs_net
def test_with_fallback_returns_semantic_when_encoder_ok(con, fake_encoder):
    result = discovery.search_indicators_with_fallback(
        "anything", fake_encoder, top_k=5, con=con
    )
    assert result.degraded is False
    # semantic mode populates similarity_score (no NULLs)
    assert result.rows["similarity_score"].null_count() == 0


@needs_net
def test_with_fallback_uses_semantic_text_as_keyword(con):
    """No keyword_query given; on failure the semantic text becomes the keyword."""
    result = discovery.search_indicators_with_fallback(
        "gdp", _BrokenEncoder(), top_k=10, con=con
    )
    assert result.degraded is True
    word = re.compile(r"\bgdp\b", re.IGNORECASE)
    for row in result.rows.iter_rows(named=True):
        haystack = (
            (row.get("indicator_code") or "")
            + " "
            + (row.get("meta_indicator_name") or "")
            + " "
            + (row.get("meta_indicator_description") or "")
        )
        assert word.search(haystack)


def test_with_fallback_wiring_offline(monkeypatch):
    """Fallback wiring without a live DB: the semantic attempt raises, the
    keyword retry returns a sentinel. Verifies the catch, the degraded flag,
    and that the retry runs keyword-only using the semantic text as the
    keyword."""
    sentinel = pl.DataFrame({"indicator_code": ["X"]})
    calls = []

    def fake_search(query, encoder, top_k, *, con, keyword_query=None):
        calls.append((query, encoder, keyword_query))
        if query:  # semantic attempt
            raise RuntimeError("encoder down")
        return sentinel  # keyword retry

    monkeypatch.setattr(discovery, "search_indicators", fake_search)

    result = discovery.search_indicators_with_fallback(
        "trade balance", object(), top_k=7, con=object()
    )
    assert result.degraded is True
    assert result.rows is sentinel
    assert len(calls) == 2  # semantic attempt, then keyword retry
    assert calls[1][0] == ""  # retry query is empty
    assert calls[1][1] is None  # retry passes no encoder
    assert calls[1][2] == "trade balance"  # keyword == the semantic text


def test_with_fallback_prefers_keyword_query_offline(monkeypatch):
    """When the caller supplied a keyword_query, the fallback uses it (not the
    semantic text)."""
    sentinel = pl.DataFrame({"indicator_code": ["X"]})
    calls = []

    def fake_search(query, encoder, top_k, *, con, keyword_query=None):
        calls.append((query, encoder, keyword_query))
        if query:
            raise RuntimeError("encoder down")
        return sentinel

    monkeypatch.setattr(discovery, "search_indicators", fake_search)

    result = discovery.search_indicators_with_fallback(
        "trade balance", object(), top_k=7, con=object(), keyword_query="LCU"
    )
    assert result.degraded is True
    assert calls[1][2] == "LCU"  # supplied keyword_query wins over semantic text
