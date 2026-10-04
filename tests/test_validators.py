"""Unit tests for the shared JSON-string parsing validator."""

from __future__ import annotations

from osaa_metrics.mcp._validators import _parse_json_string


def test_parse_json_string_parses_dict_string():
    assert _parse_json_string('{"a": 1}') == {"a": 1}


def test_parse_json_string_parses_list_string():
    assert _parse_json_string("[1, 2]") == [1, 2]


def test_parse_json_string_passes_through_non_json_string():
    assert _parse_json_string("hello") == "hello"


def test_parse_json_string_passes_through_non_string_dict():
    assert _parse_json_string({"a": 1}) == {"a": 1}


def test_parse_json_string_passes_through_non_string_list():
    assert _parse_json_string([1, 2]) == [1, 2]


def test_parse_json_string_passes_through_none():
    assert _parse_json_string(None) is None


def test_parse_json_string_passes_through_invalid_json_string():
    """An invalid-JSON string must pass through unchanged (not raise)."""
    assert _parse_json_string("{not valid json}") == "{not valid json}"
