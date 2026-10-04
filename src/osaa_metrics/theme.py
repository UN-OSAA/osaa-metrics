"""Process-level altair theme registration.

Registering the theme in the substrate keeps chart styling out of the MCP
layer: ``query_and_chart`` renders through BSL's altair backend, so a
single registration covers every chart that tool draws afterwards.

Idempotent: safe to call more than once. The branding itself lives in
``osaa_metrics/_themes/osaa_default.json`` — edit that JSON to change the
look without touching Python."""

from __future__ import annotations

import json
from importlib.resources import files

_THEME_NAME = "osaa_default"


def register_theme(name: str = _THEME_NAME) -> None:
    """Register and enable the named theme. Idempotent.

    Call once at process startup, before any chart is rendered;
    ``osaa_metrics.mcp.build_server`` does so for the MCP server."""
    import altair as alt

    spec = json.loads(
        files("osaa_metrics._themes").joinpath(f"{name}.json").read_text()
    )
    alt.theme.register(name, enable=True)(lambda: spec)
