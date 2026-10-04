"""JSON Schema-only tests for BSL semantic-model YAML.

Validates the structural schema in isolation — no BSL, no DuckDB. The schema
is the agent-facing contract and the first gate; the engine behind it (BSL
from_config + expression invocation) is covered in test_load_semantic_model.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from osaa_metrics.mcp.widget_loader import load_schema_text


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(load_schema_text("bsl_model.schema.json"))


@pytest.fixture(scope="module")
def validator(schema: dict) -> Draft202012Validator:
    return Draft202012Validator(schema)


def _errors(validator: Draft202012Validator, doc: dict) -> list[str]:
    return [e.message for e in validator.iter_errors(doc)]


def test_minimal_valid_model_passes(validator: Draft202012Validator) -> None:
    doc = yaml.safe_load(
        """
        my_model:
          table: core_tbl
          measures:
            avg_x:
              expr: _.x.mean()
        """
    )
    assert _errors(validator, doc) == []


def test_short_form_dimensions_and_measures_pass(
    validator: Draft202012Validator,
) -> None:
    doc = yaml.safe_load(
        """
        my_model:
          table: core_tbl
          dimensions:
            year: _.year
          measures:
            total_gdp: _.gdp.sum()
        """
    )
    assert _errors(validator, doc) == []


def test_long_form_with_descriptions_passes(validator: Draft202012Validator) -> None:
    doc = yaml.safe_load(
        """
        my_model:
          table: core_tbl
          dimensions:
            region:
              expr: _.region
              description: World Bank region
          measures:
            avg_gdp_growth:
              expr: _.gdp_growth.mean()
              description: Avg GDP growth (annual %)
        """
    )
    assert _errors(validator, doc) == []


def test_example_yaml_files_pass(validator: Draft202012Validator) -> None:
    """The core-model YAMLs of the worked examples under ``examples/``
    validate against the schema."""
    examples = Path(__file__).resolve().parents[1] / "examples"
    for rel in (
        "electricity_poverty/electricity_poverty.yaml",
        "economic_transformation/wdi_economy.yaml",
    ):
        doc = yaml.safe_load((examples / rel).read_text())
        errs = _errors(validator, doc)
        assert errs == [], f"{rel} failed schema: {errs}"


def test_missing_table_field_rejected(validator: Draft202012Validator) -> None:
    doc = {"my_model": {"measures": {"x": "_.x.sum()"}}}
    errs = _errors(validator, doc)
    assert any("'table' is a required property" in e for e in errs)


def test_wrong_table_value_rejected(validator: Draft202012Validator) -> None:
    """Schema pins ``table`` to the literal ``core_tbl`` — the core table handle
    name — so a model cannot point anywhere else."""
    doc = {"my_model": {"table": "some_other_tbl", "measures": {"x": "_.x.sum()"}}}
    errs = _errors(validator, doc)
    assert any("core_tbl" in e for e in errs)


def test_expr_wrong_type_rejected(validator: Draft202012Validator) -> None:
    doc = {"my_model": {"table": "core_tbl", "measures": {"x": 123}}}
    errs = _errors(validator, doc)
    assert errs, "expected schema rejection of non-string expr"


def test_lambda_expression_passes_schema(validator: Draft202012Validator) -> None:
    """An expression is any non-empty string as far as the schema is concerned,
    so a lambda clears the structural check here. Expression safety — including
    BSL's own restrictions on lambda argument names and bodies — is enforced by
    BSL's safe-eval at load time (Phase B), not by this schema."""
    doc = {"my_model": {"table": "core_tbl", "measures": {"x": "lambda _: _.x.sum()"}}}
    assert _errors(validator, doc) == []


def test_invalid_identifier_rejected(validator: Draft202012Validator) -> None:
    """Dimension/measure names must be snake_case identifiers (same rule as var_name)."""
    doc = {
        "my_model": {
            "table": "core_tbl",
            "dimensions": {"BadName": "_.x"},
        }
    }
    errs = _errors(validator, doc)
    assert errs, "expected schema rejection of non-snake-case identifier"


def test_unknown_top_level_key_rejected(validator: Draft202012Validator) -> None:
    """The model body sets ``additionalProperties: false``, so a key the schema
    does not define — ``joins`` here — is rejected instead of reaching BSL."""
    doc = {
        "my_model": {
            "table": "core_tbl",
            "joins": {"some_alias": {"model": "other"}},
        }
    }
    errs = _errors(validator, doc)
    assert errs, "expected schema rejection of unsupported keys (joins/database/filter)"


def test_two_models_rejected(validator: Draft202012Validator) -> None:
    """``maxProperties: 1`` — a core model YAML declares exactly one model."""
    doc = {
        "model_a": {"table": "core_tbl"},
        "model_b": {"table": "core_tbl"},
    }
    errs = _errors(validator, doc)
    assert errs, "expected schema rejection of multi-model document"


def test_empty_document_rejected(validator: Draft202012Validator) -> None:
    """minProperties=1 — must contain at least one model."""
    errs = _errors(validator, {})
    assert errs, "expected schema rejection of empty document"
