"""Substrate session module — round-trip + validation."""

from __future__ import annotations

import json

import pytest

from osaa_metrics.session import (
    Session,
    SessionValidationError,
    dump_session,
    load_session,
    validate_session,
)


def _minimal_session_dict() -> dict:
    return {
        "schema_version": "1",
        "generated_at": "2026-05-20T14:00:00Z",
        "schema": [
            {
                "code": "WB.GDP.GROWTH",
                "description": "GDP growth",
                "var_name": "gdp_growth",
            },
        ],
        "yaml": None,
        "queries": [],
    }


def test_round_trip_minimal_session():
    src = _minimal_session_dict()
    text = json.dumps(src)
    loaded = load_session(text)
    assert isinstance(loaded, Session)
    assert loaded.schema_version == "1"
    assert loaded.schema[0].var_name == "gdp_growth"
    redumped = dump_session(loaded)
    assert json.loads(redumped) == src


def test_rejects_unknown_schema_version():
    data = _minimal_session_dict() | {"schema_version": "2"}
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "unknown schema_version" in str(exc.value)


def test_rejects_invalid_var_name():
    data = _minimal_session_dict()
    data["schema"][0]["var_name"] = "Bad-Var"
    with pytest.raises(SessionValidationError):
        load_session(json.dumps(data))


def test_rejects_duplicate_var_names():
    data = _minimal_session_dict()
    data["schema"].append(data["schema"][0].copy())
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "duplicate" in str(exc.value)


def test_rejects_non_monotonic_query_ids():
    data = _minimal_session_dict()
    data["queries"] = [
        {
            "id": "q-002",
            "label": None,
            "created_at": "2026-05-20T14:00:00Z",
            "query_args": {},
            "chart_spec": {},
        },
        {
            "id": "q-001",
            "label": None,
            "created_at": "2026-05-20T14:01:00Z",
            "query_args": {},
            "chart_spec": {},
        },
    ]
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "not monotonic" in str(exc.value)


def test_validate_session_returns_errors_without_raising():
    errors = validate_session({"queries": "wrong"})
    assert len(errors) > 0
    assert all("path" in e and "message" in e for e in errors)


def test_query_chart_pair_round_trips():
    data = _minimal_session_dict()
    data["queries"] = [
        {
            "id": "q-001",
            "label": "Literacy by region",
            "created_at": "2026-05-20T14:00:00Z",
            "query_args": {"dimensions": ["region"], "measures": ["avg_literacy"]},
            "chart_spec": {"mark": "bar"},
            "sql": None,
        }
    ]
    loaded = load_session(json.dumps(data))
    assert loaded.queries[0].id == "q-001"
    assert loaded.queries[0].label == "Literacy by region"
    assert json.loads(dump_session(loaded)) == data


# --- Schema-driven contract guards: payload shapes the JSON Schema pass
#     must refuse before ``_session_from_dict`` ever sees them. ---


def test_rejects_query_missing_created_at():
    """A query missing `created_at` is refused during validation, so it never
    reaches `_session_from_dict` and crashes there."""
    data = _minimal_session_dict()
    data["queries"] = [
        {"id": "q-001", "query_args": {}, "chart_spec": {}},
    ]
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "created_at" in str(exc.value)


def test_rejects_query_missing_query_args():
    data = _minimal_session_dict()
    data["queries"] = [
        {"id": "q-001", "created_at": "2026-05-20T14:00:00Z", "chart_spec": {}},
    ]
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "query_args" in str(exc.value)


def test_rejects_query_missing_chart_spec():
    data = _minimal_session_dict()
    data["queries"] = [
        {"id": "q-001", "created_at": "2026-05-20T14:00:00Z", "query_args": {}},
    ]
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "chart_spec" in str(exc.value)


def test_rejects_additional_top_level_field():
    """The session schema sets `additionalProperties: false` and load_session
    enforces it, so an unknown top-level field is a load error rather than
    silently carried data."""
    data = _minimal_session_dict()
    data["secret_backdoor"] = "ignored"  # noqa: S105  # a fixture key, not a credential
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "secret_backdoor" in str(exc.value) or "additional" in str(exc.value).lower()


def test_rejects_additional_field_in_schema_row():
    data = _minimal_session_dict()
    data["schema"][0]["color"] = "red"
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "color" in str(exc.value) or "additional" in str(exc.value).lower()


def test_rejects_non_iso_generated_at():
    """`generated_at` is checked against `format: date-time`, so a free-text
    timestamp is refused instead of reaching a consumer that parses it."""
    data = _minimal_session_dict()
    data["generated_at"] = "today at noon"
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "generated_at" in str(exc.value)


def test_accepts_both_z_and_offset_iso_datetimes():
    """Both `2026-05-20T14:00:00Z` and `2026-05-20T14:00:00+00:00` are
    valid RFC 3339 / ISO 8601 — the validator must accept both shapes since
    our own writers use the offset form (`datetime.isoformat(timespec=...)`)."""
    for ts in ("2026-05-20T14:00:00Z", "2026-05-20T14:00:00+00:00"):
        data = _minimal_session_dict() | {"generated_at": ts}
        # Should not raise.
        load_session(json.dumps(data))


def test_duplicate_query_id_caught():
    """Cross-row invariant: each query id must be unique. JSON Schema can't
    easily express this, so the Python pass catches it."""
    data = _minimal_session_dict()
    data["queries"] = [
        {
            "id": "q-001",
            "label": None,
            "created_at": "2026-05-20T14:00:00Z",
            "query_args": {},
            "chart_spec": {},
        },
        {
            "id": "q-001",
            "label": None,
            "created_at": "2026-05-20T14:01:00Z",
            "query_args": {},
            "chart_spec": {},
        },
    ]
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "duplicate" in str(exc.value)


def test_rejects_naive_datetime_without_offset():
    """Naive ISO datetimes (no offset) are RFC 3339-invalid and round-trip
    ambiguously. Our own writers always emit an offset via
    ``datetime.now(timezone.utc).isoformat`` — incoming session JSON must
    match that contract."""
    data = _minimal_session_dict() | {"generated_at": "2026-05-22T14:00:00"}
    with pytest.raises(SessionValidationError) as exc:
        load_session(json.dumps(data))
    assert "generated_at" in str(exc.value)


def test_validate_session_collects_multiple_errors_in_one_pass():
    """jsonschema enumerates ALL errors, not just the first one — so the
    agent gets the full punch list per call instead of fix-one-reveal-next."""
    bad = {
        "schema_version": "2",  # wrong const
        "generated_at": "nope",  # bad format
        "schema": "wrong",  # wrong type
        "queries": [],
    }
    errors = validate_session(bad)
    paths = {e["path"] for e in errors}
    # Schema_version, generated_at, and schema all need to be flagged together.
    assert "schema_version" in paths
    assert "generated_at" in paths
    assert "schema" in paths


def test_session_roundtrip_preserves_sql():
    from osaa_metrics.session import QueryChartPair, Session, dump_session, load_session

    s = Session(
        schema_version="1",
        generated_at="2026-06-01T00:00:00+00:00",
        schema=[],
        yaml=None,
        queries=[
            QueryChartPair(
                id="q-001",
                label=None,
                created_at="2026-06-01T00:00:00+00:00",
                query_args={"dimensions": ["region"]},
                chart_spec={"mark": "bar"},
                sql="SELECT region FROM core",
            )
        ],
    )
    loaded = load_session(dump_session(s))
    assert loaded.queries[0].sql == "SELECT region FROM core"


def test_load_session_accepts_legacy_without_sql():
    """A session JSON whose query/chart pairs omit `sql` still loads — the
    field is optional and defaults to None."""
    from osaa_metrics.session import load_session

    legacy = (
        '{"schema_version":"1","generated_at":"2026-06-01T00:00:00+00:00",'
        '"schema":[],"yaml":null,'
        '"queries":[{"id":"q-001","created_at":"2026-06-01T00:00:00+00:00",'
        '"query_args":{},"chart_spec":{}}]}'
    )
    assert load_session(legacy).queries[0].sql is None
