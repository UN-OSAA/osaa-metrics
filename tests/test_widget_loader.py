"""Tests for widget_loader: relative module imports must be inlined at serve time
because MCP-app iframes don't resolve sibling ui:// URLs."""

from __future__ import annotations

from pathlib import Path

import pytest

from osaa_metrics.mcp import widget_loader


@pytest.fixture
def fake_widgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(widget_loader, "widgets_dir", lambda name: tmp_path)
    return tmp_path


def test_load_widget_inlines_relative_import(fake_widgets: Path) -> None:
    (fake_widgets / "shared.js").write_text(
        "export function hello() { return 'hi'; }\nexport const X = 42;\n"
    )
    (fake_widgets / "view.html").write_text(
        '<script type="module">\n'
        'import { hello, X } from "./shared.js";\n'
        "console.log(hello(), X);\n"
        "</script>\n"
    )
    out = widget_loader.load_widget("view.html")
    assert 'import { hello, X } from "./shared.js"' not in out
    assert "function hello()" in out
    assert "const X = 42" in out
    # Top-level export keyword stripped so inlined code is valid in the surrounding module.
    assert "export function" not in out
    assert "export const" not in out
    # Surrounding HTML still present.
    assert '<script type="module">' in out
    assert "console.log(hello(), X);" in out


def test_load_widget_preserves_absolute_imports(fake_widgets: Path) -> None:
    (fake_widgets / "view.html").write_text(
        '<script type="module">\n'
        'import { App } from "https://unpkg.com/@modelcontextprotocol/ext-apps@1.7.1/app-with-deps";\n'
        "</script>\n"
    )
    out = widget_loader.load_widget("view.html")
    assert "https://unpkg.com/@modelcontextprotocol/ext-apps@1.7.1/app-with-deps" in out


def test_load_widget_real_discovery_inlines_shared_js() -> None:
    """The real discovery.html (shipped in mcp_discovery/_widgets/) must come back self-contained."""
    out = widget_loader.load_widget("discovery.html")
    assert 'from "./shared.js"' not in out
    # Functions from shared.js end up inlined.
    assert "function parseToolResult" in out
    assert "function nowSlug" in out
    assert "function validateVarName" in out
    assert "function toCSV" in out
    # The CSV download goes through the host-mediated app.downloadFile path,
    # so shared.js carries no blob-URL download helper.
    assert "function triggerDownload" not in out
    # Absolute import stays.
    assert "https://unpkg.com/@modelcontextprotocol/ext-apps@1.7.1/app-with-deps" in out


def test_load_widget_real_core_summary_inlines_parse_helper() -> None:
    """core_summary.html imports parseToolResult from shared.js, so the served
    document must carry that helper inlined."""
    out = widget_loader.load_widget("core_summary.html")
    assert 'from "./shared.js"' not in out
    assert "function parseToolResult" in out


def test_load_widget_real_core_summary_serves_clean() -> None:
    """The core-summary widget has no relative imports but must still load through load_widget."""
    out = widget_loader.load_widget("core_summary.html")
    assert "Core table — summary" in out
    assert "https://unpkg.com/@modelcontextprotocol/ext-apps@1.7.1/app-with-deps" in out


def test_terminology_lock_ins_in_discovery_widget() -> None:
    """Terminology lock-in: button labels and status vocab must use 'Save'/'saved', not 'sync'."""
    html = widget_loader.load_widget("discovery.html")
    assert ">Save<" in html
    assert ">Download (.csv)<" in html
    # The forbidden substitutes must be absent from user-visible labels.
    assert "Use these indicators" not in html
    assert "Download snapshot" not in html
    assert "synced" not in html
    assert "edited since last sync" not in html


# Forbidden substitutes for "core semantic table" (the vocabulary table in
# skills/osaa-metrics/SKILL.md). The test guards against the
# bigram substitutes only — single tokens like "model" remain legitimate in
# domain prose like "BSL semantic model".
_CORE_SEMANTIC_TABLE_SUBSTITUTES = (
    "loaded model",
    "model object",
)


@pytest.mark.parametrize(
    "widget", ["discovery.html", "chart.html", "core_summary.html"]
)
def test_no_core_semantic_table_substitutes(widget: str) -> None:
    """Every widget's served HTML must be free of the two-word substitutes for
    'core semantic table' that the skill's vocabulary table forbids."""
    html = widget_loader.load_widget(widget)
    for bad in _CORE_SEMANTIC_TABLE_SUBSTITUTES:
        assert bad not in html, f"{widget!r} contains forbidden substitute {bad!r}"


def test_chart_widget_has_four_tabs():
    from osaa_metrics.mcp.widget_loader import load_widget

    html = load_widget("chart.html")
    for pane in ["chart", "query", "sql", "chartspec"]:
        assert f'data-pane="{pane}"' in html, f"missing tab pane {pane}"
    assert "chart_error" not in html  # no chart_error field for the widget to render


def test_discovery_searches_on_enter_not_keystroke():
    from osaa_metrics.mcp.widget_loader import load_widget

    html = load_widget("discovery.html")
    # The search boxes don't fire on every keystroke...
    assert 'semBox.addEventListener("input"' not in html
    assert 'kwBox.addEventListener("input"' not in html
    assert "debouncedSearch" not in html
    # ...they submit on Enter.
    assert "keydown" in html
    assert "press Enter to search" in html


def test_chart_widget_has_no_download_buttons():
    from osaa_metrics.mcp.widget_loader import load_widget

    html = load_widget("chart.html")
    for marker in [
        "dl-yaml",
        "dl-session",
        "offerDownload",
        "fetchSessionPayload",
        "enableYamlButton",
        "downloadFile",
    ]:
        assert marker not in html, f"chart widget still references {marker}"


def test_shared_js_byte_identical_across_packages() -> None:
    """``mcp/_widgets/shared.js`` and ``mcp_discovery/_widgets/shared.js`` must
    stay byte-identical while each sub-package carries its own copy, kept in
    sync by hand. The widget inliner picks one file per sub-package; silent drift
    would only surface in production when a widget loaded from the wrong
    one. This test catches a single-side edit immediately."""
    from importlib.resources import files

    a = (files("osaa_metrics.mcp") / "_widgets" / "shared.js").read_text(
        encoding="utf-8"
    )
    b = (files("osaa_metrics.mcp_discovery") / "_widgets" / "shared.js").read_text(
        encoding="utf-8"
    )
    assert a == b, "shared.js files have drifted; keep the two copies byte-identical"
