"""Substrate altair theme registration."""

import altair as alt

from osaa_metrics.theme import register_theme


def test_register_theme_enables_osaa_default():
    register_theme()
    assert alt.theme.active == "osaa_default"


def test_register_theme_is_idempotent():
    register_theme()
    register_theme()
    assert alt.theme.active == "osaa_default"


def test_theme_spec_loads_and_has_config():
    register_theme()
    spec = alt.theme.get()()
    assert isinstance(spec, dict)
    assert "config" in spec
