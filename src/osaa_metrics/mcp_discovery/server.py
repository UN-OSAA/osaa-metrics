"""Discovery MCP sub-server — search + working set commit.

OOP pattern modeled on BSL's ``MCPSemanticModel``
(``boring_semantic_layer/agents/backends/mcp.py`` in the installed package).
Subclasses ``FastMCP``, takes deps via constructor, registers tools as inner
functions in ``_register_tools()`` that close over ``self``.

CONTRACT: zero BSL imports anywhere in this sub-package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.apps import AppConfig, ResourceCSP
from pydantic import Field
from pydantic.functional_validators import BeforeValidator

from osaa_metrics.mcp import session_state
from osaa_metrics.mcp import tools as composing_tools
from osaa_metrics.mcp._validators import _parse_json_string
from osaa_metrics.mcp.providers import DataSourceProvider, EncoderProvider
from osaa_metrics.mcp.widget_loader import load_widget
from osaa_metrics.mcp_discovery.tools import (
    handle_open_discovery,
    handle_save_core_schema,
)

if TYPE_CHECKING:
    import ibis


_DISCOVERY_URI = "ui://osaa-metrics/discovery.html"

_DISCOVERY_INSTRUCTIONS = (
    "Indicator discovery — semantic + keyword search over the indicator catalogue. "
    "The agent should call open_discovery, then call build_core_table "
    "when the user says they saved; "
    "discover_indicators is widget plumbing, not for direct calls."
)


class DiscoveryMCPServer(FastMCP):
    """FastMCP subclass exposing the indicator discovery surface.

    Stays BSL-free: the sub-package imports nothing from
    ``boring_semantic_layer``. It holds and mutates ``models``, the registry
    of core semantic tables, but only as an opaque dict it never introspects
    — this package only ever pops this session's keys. BSL enters one level up, in the
    composing ``OSAAMetricsServer``, which imports ``MCPSemanticModel`` and
    mounts it.
    """

    def __init__(
        self,
        data_source: DataSourceProvider,
        encoder_provider: EncoderProvider,
        models: dict,
        name: str = "osaa-discovery",
        instructions: str = _DISCOVERY_INSTRUCTIONS,
        **kwargs: Any,
    ):
        super().__init__(name=name, instructions=instructions, **kwargs)
        self.data_source = data_source
        self.encoder_provider = encoder_provider
        self.models = models
        self._register_tools()
        self._register_resources()

    @property
    def con(self) -> ibis.BaseBackend:
        """Lazy backend — raises a named ToolValidationError inside tool calls
        when the data source is unreachable (graceful boot)."""
        return self.data_source.get_con()

    def _register_tools(self) -> None:
        @self.tool(
            name="open_discovery",
            description=(
                "Open the indicator-discovery widget. Pass the user's discovery intent "
                "as initial_semantic_query so the widget opens already populated; "
                "initial_keyword_query seeds the keyword filter. The widget runs in an "
                "iframe and calls save_core_schema when the user clicks Save; "
                "that reply goes to the widget. The state this tool returns is "
                "read when the tool runs, before any Save in the widget it opens."
            ),
            app=AppConfig(resource_uri=_DISCOVERY_URI),
        )
        async def open_discovery(
            ctx: Context,
            initial_semantic_query: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Optional seed for the semantic search input",
                ),
            ] = None,
            initial_keyword_query: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Optional seed for the keyword filter input",
                ),
            ] = None,
        ) -> dict:
            return await handle_open_discovery(
                ctx, initial_semantic_query, initial_keyword_query
            )

        @self.tool(
            name="discover_indicators",
            description=(
                "Widget plumbing — search the indicator catalogue. Do NOT call directly; "
                "the discovery widget calls this via app.callServerTool."
            ),
        )
        async def discover_indicators(
            ctx: Context,
            semantic_query: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Semantic query (empty → keyword-only or browse)",
                ),
            ] = None,
            keyword_query: Annotated[
                str | None,
                Field(
                    default=None, description="Optional word-boundary keyword filter"
                ),
            ] = None,
            top_k: Annotated[
                int,
                Field(
                    default=50,
                    ge=1,
                    le=composing_tools.TOP_K_MAX,
                    description="Maximum rows to return",
                ),
            ] = 50,
        ) -> dict:
            semantic = (semantic_query or "").strip()
            note: str | None = None
            degraded = False
            encoder = None
            if semantic:
                # Only touch the encoder provider when there's a semantic
                # query — keyword browse must never trigger a model load.
                try:
                    encoder = self.encoder_provider.get()
                except Exception:  # noqa: BLE001  # the encoder floor: a failure to build the encoder degrades to keyword search
                    note = "ranking unavailable"
                    degraded = True
            if semantic and encoder is None:
                # The encoder could not be built: run the same text as a
                # keyword search.
                rows, _ = composing_tools.discover_indicators(
                    None,
                    keyword_query if (keyword_query or "").strip() else semantic,
                    top_k,
                    con=self.con,
                    encoder=None,
                )
            else:
                rows, fell_back = composing_tools.discover_indicators(
                    semantic_query,
                    keyword_query,
                    top_k,
                    con=self.con,
                    encoder=encoder,
                )
                if fell_back:
                    degraded = True
                    note = "ranking unavailable"
            # A query the composing tool rejects raises before this line, so it
            # never reaches session_state, whichever branch above ran it.
            await session_state.set_queries(
                ctx, semantic_query or "", keyword_query or ""
            )
            return {"candidates": rows, "degraded": degraded, "note": note}

        @self.tool(
            name="save_core_schema",
            description=(
                "Save {code, description, var_name} rows as the core table schema. "
                "The discovery widget's Save button calls this. Call it yourself "
                "only when the user attaches a core table schema CSV with code, "
                "description and var_name columns."
            ),
        )
        async def save_core_schema(
            ctx: Context,
            working_set: Annotated[
                list[dict],
                BeforeValidator(_parse_json_string),
                Field(
                    description=(
                        "List of {code, description, var_name} rows to save as the "
                        "core table schema"
                    )
                ),
            ],
        ) -> dict:
            return await handle_save_core_schema(ctx, working_set, models=self.models)

    def _register_resources(self) -> None:
        @self.resource(
            _DISCOVERY_URI,
            app=AppConfig(csp=ResourceCSP(resource_domains=["https://unpkg.com"])),
        )
        def discovery_view() -> str:
            return load_widget("discovery.html")
