"""Shared pydantic validator for JSON-stringified dict/list parameters."""

import json
from typing import Any


def _parse_json_string(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    return v
