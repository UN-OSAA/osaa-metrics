"""Schema rows are capped at Save (100, mirrored by the session JSON
Schema's maxItems) — the pivot's own width limit. Query/chart pairs carry
no count cap: every define_semantic_model call and every breaking Save
already clears the whole analysis history, so only pairs matching the
current core table and core-model YAML ever survive — a count ceiling
protected nothing semantic. The reload ceiling (SESSION_JSON_MAX) is sized
with headroom over the schema cap at typical field sizes."""

from osaa_metrics.mcp.tools import (
    CORE_TABLE_SCHEMA_ROWS_MAX,
    SESSION_JSON_MAX,
    handle_get_session_schema,
)
from osaa_metrics.session import (
    SCHEMA_VERSION,
    QueryChartPair,
    SchemaRow,
    Session,
    dump_session,
    validate_session,
)


def _doc(n_schema=1, n_queries=0):
    return {
        "schema_version": "1",
        "generated_at": "2026-08-26T00:00:00+00:00",
        "schema": [
            {"code": f"C{i}", "description": "d", "var_name": f"v_{i}"}
            for i in range(n_schema)
        ],
        "yaml": None,
        "queries": [
            {
                "id": f"q-{i + 1:03d}",
                "label": "l",
                "created_at": "2026-08-26T00:00:00+00:00",
                "query_args": {},
                "chart_spec": {},
                "sql": None,
            }
            for i in range(n_queries)
        ],
    }


def test_schema_maxitems_100():
    assert validate_session(_doc(n_schema=100)) == []
    errors = validate_session(_doc(n_schema=101))
    assert errors and any(
        e["path"] == "schema" and e["message"].endswith("is too long") for e in errors
    )


def test_maximal_count_capped_session_reloads_under_ceiling():
    """A session at the schema's 100-row maxItems ceiling, carrying 100
    query/chart pairs as an illustrative volume (queries has no count
    ceiling), with realistic per-field content, validates against the
    session schema and serializes to fewer bytes than SESSION_JSON_MAX.
    Covers the count dimension only — per-field size (an unbounded filter
    value list, say) is a separate, unbounded hole this test does not
    exercise."""
    schema_rows = [
        SchemaRow(
            code=f"NY.GDP.MKTP.CD.{i:03d}",
            description=(
                f"Gross domestic product at market prices, current US dollars, series {i:03d}"
            ),
            var_name=f"gdp_{i:03d}",
        )
        for i in range(100)
    ]
    queries = [
        QueryChartPair(
            id=f"q-{i + 1:03d}",
            label=f"GDP trend by region, series {i:03d}, 2010-2020",
            created_at="2026-08-26T00:00:00+00:00",
            query_args={
                "dimensions": ["region", "year"],
                "measures": [f"gdp_{i:03d}"],
                "filters": [{"field": "year", "op": ">=", "value": 2010}],
                "order_by": [["year", "asc"]],
                "limit": 1000,
            },
            chart_spec={
                "mark": "line",
                "encoding": {
                    "x": {"field": "year", "type": "temporal"},
                    "y": {"field": f"gdp_{i:03d}", "type": "quantitative"},
                    "color": {"field": "region", "type": "nominal"},
                },
            },
            sql=(
                f'SELECT "region", "year", AVG("gdp_{i:03d}") AS "avg_gdp_{i:03d}" '  # noqa: S608  # stored as session data, never executed
                f'FROM "core_tbl" WHERE "year" >= 2010 GROUP BY "region", "year" '
                f'ORDER BY "year" ASC LIMIT 1000'
            ),
        )
        for i in range(100)
    ]
    session = Session(
        schema_version="1",
        generated_at="2026-08-26T00:00:00+00:00",
        schema=schema_rows,
        yaml=None,
        queries=queries,
    )
    content = dump_session(session)
    assert validate_session(content) == []
    assert len(content.encode("utf-8")) < SESSION_JSON_MAX


def test_save_cap_and_schema_version_match_the_session_json_schema():
    """The JSON Schema cannot import Python constants, so this test is what
    keeps the Save cap and the schema version in code equal to the schema's
    own ``maxItems`` and ``schema_version`` const."""
    schema = handle_get_session_schema()
    assert schema["properties"]["schema"]["maxItems"] == CORE_TABLE_SCHEMA_ROWS_MAX
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
