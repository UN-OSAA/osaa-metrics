"""Semantic helpers: label resolution falls back across BSL → yaml dict → name."""

from osaa_metrics.semantic import display_label


class _StubField:
    def __init__(self, description=None):
        self.description = description


class _StubModel:
    def __init__(self, dims, measures):
        self._dims = dims
        self._measures = measures

    def get_measures(self):
        return self._measures

    def get_dimensions(self):
        return self._dims


def test_display_label_prefers_measure_description():
    model = _StubModel(
        dims={"region": _StubField("Region")},
        measures={"gdp": _StubField("Total GDP (USD)")},
    )
    assert display_label(model, "gdp") == "Total GDP (USD)"


def test_display_label_falls_back_to_yaml_descriptions():
    model = _StubModel(
        dims={},
        measures={"gdp": _StubField(None)},
    )
    yaml_d = {"gdp": "From YAML"}
    assert display_label(model, "gdp", yaml_descriptions=yaml_d) == "From YAML"


def test_display_label_falls_back_to_raw_name():
    model = _StubModel(dims={}, measures={"x": _StubField(None)})
    assert display_label(model, "x") == "x"


def test_bsl_schema_validator_is_built_once() -> None:
    """The schema is parsed and the validator built once per process."""
    from osaa_metrics.semantic import _bsl_schema_validator

    assert _bsl_schema_validator() is _bsl_schema_validator()
