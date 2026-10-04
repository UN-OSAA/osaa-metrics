"""osaa-metrics — chat-native pair-analyst library for OSAA economic indicators.

Re-exports the curated public surface — discovery search, the core table
build, core model YAML validation, the core summary renderer, the portable
session artifact, the settings/connection helpers, and the chart theme; ``__all__`` below is
the surface itself. The MCP server lives in ``osaa_metrics.mcp`` behind the
``[mcp]`` extra and is not imported here, so importing the library never
pulls FastMCP into the process."""

from osaa_metrics.config import Settings, build_connection, load_settings
from osaa_metrics.core_table import (
    assign_var_names,
    build_wide_table,
    enrich_core_table,
    load_from_csv,
    slugify_indicator_name,
)
from osaa_metrics.discovery import (
    SearchResult,
    load_embedding_cache,
    search_indicators,
    search_indicators_with_fallback,
)
from osaa_metrics.semantic import (
    ModelValidationError,
    display_label,
    summarize_model,
    validate_and_load_model,
)
from osaa_metrics.session import (
    Session,
    SessionValidationError,
    dump_session,
    load_session,
    validate_session,
)
from osaa_metrics.summarize import build_core_summary_payload_from_handle
from osaa_metrics.theme import register_theme

__version__ = "0.1.0"

__all__ = [
    "ModelValidationError",
    "SearchResult",
    "Session",
    "SessionValidationError",
    "Settings",
    "__version__",
    "assign_var_names",
    "build_connection",
    "build_core_summary_payload_from_handle",
    "build_wide_table",
    "display_label",
    "dump_session",
    "enrich_core_table",
    "load_embedding_cache",
    "load_from_csv",
    "load_session",
    "load_settings",
    "register_theme",
    "search_indicators",
    "search_indicators_with_fallback",
    "slugify_indicator_name",
    "summarize_model",
    "validate_and_load_model",
    "validate_session",
]
