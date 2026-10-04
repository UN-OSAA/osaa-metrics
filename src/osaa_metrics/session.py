"""Portable session bootstrap — the recipe artifact any caller can load.

The session JSON is the canonical "continue this work" artifact downloaded
from chat. The agent (via the load_session MCP tool) and any other caller
parse it through this module.

Trust principle: validate every input regardless of provenance. Every
caller's input goes through identical validation. The JSON Schema at
``_schemas/session.schema.json`` is the single source of truth — both this
validator and the ``get_session_schema`` resource read from the same file."""

from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

SCHEMA_VERSION = "1"
_SCHEMA = files("osaa_metrics").joinpath("_schemas", "session.schema.json")


_format_checker = FormatChecker()


@_format_checker.checks("date-time", ValueError)
def _check_iso_datetime(value: Any) -> bool:
    """Accept RFC 3339 / ISO 8601 date-times that carry an explicit offset.

    Python's ``fromisoformat`` accepts both a trailing ``Z`` and an explicit
    offset like ``+00:00``. We additionally require that an offset be
    present — naive datetimes (no timezone) round-trip
    ambiguously through ``download_session`` / ``load_session`` and our own
    writers always emit an offset via ``datetime.now(timezone.utc).isoformat``."""
    if not isinstance(value, str):
        return True  # jsonschema's `type` check handles non-strings.
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(
            "date-time must carry an explicit offset (e.g. '+00:00' or 'Z')"
        )
    return True


@functools.lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    """Cached jsonschema validator built from the packaged schema file."""
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=_format_checker)


class SessionValidationError(ValueError):
    """Raised by load_session on any structural or referential error."""

    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        super().__init__(json.dumps({"errors": errors}))


@dataclass
class SchemaRow:
    """One core table schema row: indicator code, description and var_name."""

    code: str
    description: str
    var_name: str


@dataclass
class QueryChartPair:
    """One query/chart pair as a session file stores it: id, label,
    timestamp, query arguments, chart spec, SQL."""

    id: str
    label: str | None
    created_at: str
    query_args: dict[str, Any]
    chart_spec: dict[str, Any]
    sql: str | None = None


@dataclass
class Session:
    """A session file in memory: the core table schema, the core-model YAML,
    and the query/chart pairs, in the shape ``_schemas/session.schema.json``
    defines."""

    schema_version: str
    generated_at: str
    schema: list[SchemaRow]
    yaml: str | None
    queries: list[QueryChartPair]


def load_session(text: str) -> Session:
    """Parse JSON, validate, return a Session. Raises SessionValidationError
    on any structural problem — the caller is expected to catch it: the MCP
    tool wraps it as a ToolError, and any other caller surfaces it itself."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SessionValidationError(
            [{"path": "", "message": f"invalid JSON: {e.msg}", "hint": ""}]
        ) from e
    errors = validate_session(data)
    if errors:
        raise SessionValidationError(errors)
    return _session_from_dict(data)


def dump_session(session: Session) -> str:
    """Serialize a Session to a JSON string.

    Keys keep their insertion order (no sorting): a chart_spec reads best in
    the order vega-lite authors write it.
    """
    return json.dumps(_session_to_dict(session))


def _path_str(err: ValidationError) -> str:
    """Render a jsonschema absolute_path as ``foo[0].bar[1].baz`` — matches
    the path shape used by the rest of the codebase's structured errors."""
    parts: list[str] = []
    for p in err.absolute_path:
        if isinstance(p, int):
            if parts:
                parts[-1] = parts[-1] + f"[{p}]"
            else:
                parts.append(f"[{p}]")
        else:
            parts.append(("." if parts else "") + str(p))
    return "".join(parts)


def _normalize_schema_error(err: ValidationError) -> dict[str, str]:
    """Convert a jsonschema ValidationError into the {path,message,hint} shape.
    A few well-known violations get hand-crafted messages so the error stays
    readable; the rest use jsonschema's default phrasing."""
    abs_path = list(err.absolute_path)

    # A schema_version mismatch gets its own message: it names the version
    # found, and the hint names the version this build reads.
    if abs_path == ["schema_version"] and err.validator == "const":
        return {
            "path": "schema_version",
            "message": f"unknown schema_version {err.instance!r}",
            "hint": f"expected {SCHEMA_VERSION!r}",
        }

    # var_name pattern violation — keep the explicit regex in the message.
    if (
        len(abs_path) == 3
        and abs_path[0] == "schema"
        and abs_path[2] == "var_name"
        and err.validator == "pattern"
    ):
        return {
            "path": _path_str(err),
            "message": f"{err.instance!r} must match ^[a-z_][a-z0-9_]*$",
            "hint": "snake_case identifier",
        }

    # query id pattern violation — same idea.
    if (
        len(abs_path) == 3
        and abs_path[0] == "queries"
        and abs_path[2] == "id"
        and err.validator == "pattern"
    ):
        return {
            "path": _path_str(err),
            "message": f"{err.instance!r} must match ^q-\\d{{3,}}$",
            "hint": "e.g. q-001",
        }

    return {"path": _path_str(err), "message": err.message, "hint": ""}


def validate_session(payload: Any) -> list[dict[str, str]]:
    """Return a list of error dicts without raising. Each error has the
    {path, message, hint} shape the rest of the codebase's structured errors
    use.

    The JSON Schema (``_schemas/session.schema.json``) is the contract;
    jsonschema enumerates the structural violations in one pass. On top of
    that we run the cross-row invariants JSON Schema can't express cleanly:
    var_name uniqueness within ``schema[]`` and monotonic ``q-NNN`` ordering
    within ``queries[]``.

    payload may be a dict (already parsed) or a string (JSON)."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            return [{"path": "", "message": f"invalid JSON: {e.msg}", "hint": ""}]
    if not isinstance(payload, dict):
        return [{"path": "", "message": "session must be a JSON object", "hint": ""}]

    validator = _schema_validator()
    schema_errs = sorted(
        validator.iter_errors(payload), key=lambda e: list(e.absolute_path)
    )
    if schema_errs:
        # If the structural contract failed, return early — the cross-row
        # invariants below assume the structure is already valid.
        return [_normalize_schema_error(e) for e in schema_errs]

    errors: list[dict[str, str]] = []

    seen_var_names: set[str] = set()
    for i, row in enumerate(payload["schema"]):
        vn = row["var_name"]
        if vn in seen_var_names:
            errors.append(
                {
                    "path": f"schema[{i}].var_name",
                    "message": f"duplicate {vn!r}",
                    "hint": "",
                }
            )
        seen_var_names.add(vn)

    seen_ids: set[str] = set()
    max_id_num = -1
    for i, q in enumerate(payload["queries"]):
        qid = q["id"]
        if qid in seen_ids:
            errors.append(
                {
                    "path": f"queries[{i}].id",
                    "message": f"duplicate {qid!r}",
                    "hint": "",
                }
            )
            continue
        seen_ids.add(qid)
        id_num = int(qid.split("-", 1)[1])
        if id_num <= max_id_num:
            errors.append(
                {
                    "path": f"queries[{i}].id",
                    "message": f"id {qid!r} not monotonic (previous max {max_id_num})",
                    "hint": "each id must be strictly greater than the previous max",
                }
            )
        max_id_num = max(max_id_num, id_num)

    return errors


def _session_from_dict(data: dict[str, Any]) -> Session:
    return Session(
        schema_version=data["schema_version"],
        generated_at=data["generated_at"],
        schema=[SchemaRow(**row) for row in data.get("schema", [])],
        yaml=data.get("yaml"),
        queries=[
            QueryChartPair(
                id=q["id"],
                label=q.get("label"),
                created_at=q["created_at"],
                query_args=q.get("query_args", {}),
                chart_spec=q.get("chart_spec", {}),
                sql=q.get("sql"),
            )
            for q in data.get("queries", [])
        ],
    )


def _session_to_dict(session: Session) -> dict[str, Any]:
    return {
        "schema_version": session.schema_version,
        "generated_at": session.generated_at,
        "schema": [asdict(row) for row in session.schema],
        "yaml": session.yaml,
        "queries": [asdict(q) for q in session.queries],
    }
