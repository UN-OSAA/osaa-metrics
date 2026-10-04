"""Resolve ui:// widget resources to HTML files shipped inside the package.

MCP Apps deliver a ui:// resource to the iframe as a single document with no
sibling-resource URL — so a relative ``import ... from "./shared.js"`` 404s
and the whole module script aborts before loading the App SDK. ``load_widget``
inlines those local imports at serve time so the HTML is self-contained on
the wire while the source files stay separate for reuse and testing.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

# Matches: import { a, b, c } from "./file.js";
# Captures: (1) braced names, (2) relative path
_LOCAL_IMPORT_RE = re.compile(
    r'^[ \t]*import\s*\{([^}]+)\}\s*from\s*"(\./[^"]+)"\s*;[ \t]*$',
    re.MULTILINE,
)
_EXPORT_PREFIX_RE = re.compile(r"^export\s+", re.MULTILINE)

# Each widget ships with the sub-package whose tools surface it. One loader
# serves both packages; this map names the owner of each file.
_WIDGET_PACKAGE = {
    "discovery.html": "osaa_metrics.mcp_discovery",
    "chart.html": "osaa_metrics.mcp",
    "core_summary.html": "osaa_metrics.mcp",
}


def widgets_dir(name: str) -> Path:
    """Return the directory containing widget `name`.

    Resolves to ``<package>/_widgets/`` shipped inside the wheel, where
    ``<package>`` is the sub-package that surfaces this widget (see
    ``_WIDGET_PACKAGE``).
    """
    pkg = _WIDGET_PACKAGE.get(name, "osaa_metrics.mcp")
    return Path(str(files(pkg) / "_widgets"))


def _inline_local_module(match: re.Match[str], base: Path) -> str:
    """Replace one ``import { ... } from "./file.js"`` line with the file's contents.

    The imported file's ``export`` keywords are stripped so the inlined code
    becomes top-level declarations in the surrounding ``<script type="module">``.
    Any local imports in the inlined file are recursively expanded.
    """
    rel = match.group(2)
    target = (base / rel).resolve()
    src = target.read_text(encoding="utf-8")
    src = _LOCAL_IMPORT_RE.sub(lambda m: _inline_local_module(m, target.parent), src)
    src = _EXPORT_PREFIX_RE.sub("", src)
    header = f"// --- inlined from {rel} ---"
    footer = f"// --- end {rel} ---"
    return f"{header}\n{src}\n{footer}"


def load_widget(name: str) -> str:
    """Load a widget HTML file by name (e.g. 'discovery.html'), inlining local module imports.

    Relative ``import { ... } from "./*.js"`` lines are replaced with the imported
    file's contents (with ``export`` stripped); absolute URLs (https://, etc.) are
    left alone. Raises FileNotFoundError if the widget HTML is missing.
    """
    path = widgets_dir(name) / name
    html = path.read_text(encoding="utf-8")
    return _LOCAL_IMPORT_RE.sub(lambda m: _inline_local_module(m, path.parent), html)


def load_schema_text(name: str) -> str:
    """Load a substrate-owned schema file (e.g. ``bsl_model.schema.json``).

    The schema lives in ``osaa_metrics/_schemas/`` so substrate modules
    (``semantic.py``) can validate without depending on the MCP sub-package's
    asset layout. The MCP server re-serves the same bytes as a resource.

    Read through the traversable so it also works from a still-zipped wheel.
    """
    return files("osaa_metrics").joinpath("_schemas", name).read_text(encoding="utf-8")
