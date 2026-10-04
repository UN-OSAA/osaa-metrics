"""Composing MCP server — mounts ``DiscoveryMCPServer`` + BSL ``MCPSemanticModel``
and registers the composing-layer tools.

Server naming follows attribution discipline:

- ``osaa-metrics`` — top-level (ours)
- ``osaa-discovery`` — mounted sub-server (ours)
- ``BSL`` — mounted ``MCPSemanticModel`` (third-party, unchanged)

OOP pattern modeled on BSL's ``MCPSemanticModel``
(``boring_semantic_layer/agents/backends/mcp.py`` in the installed package).
Constructor-DI for ``data_source`` / ``encoder_provider`` / ``models``; no
module-level singletons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from boring_semantic_layer import MCPSemanticModel
from fastmcp import Context, FastMCP
from fastmcp.apps import AppConfig, ResourceCSP
from pydantic import Field
from pydantic.functional_validators import BeforeValidator
from starlette.responses import PlainTextResponse, Response

from osaa_metrics.mcp._validators import _parse_json_string
from osaa_metrics.mcp.providers import DataSourceProvider, EncoderProvider
from osaa_metrics.mcp.tools import (
    handle_build_core_summary,
    handle_build_core_table,
    handle_define_semantic_model,
    handle_download_session,
    handle_get_bsl_schema,
    handle_get_session_schema,
    handle_load_session,
    handle_prepare_download,
    handle_query_and_chart,
    handle_reset_state,
)
from osaa_metrics.mcp.widget_loader import load_widget
from osaa_metrics.mcp_discovery import DiscoveryMCPServer

if TYPE_CHECKING:
    import ibis


_CHART_URI = "ui://osaa-metrics/chart.html"
_CORE_SUMMARY_URI = "ui://osaa-metrics/core_summary.html"
_BSL_SCHEMA_URI = "bsl-schema://osaa-metrics/v1"
_SESSION_SCHEMA_URI = "session-schema://osaa-metrics/v1"

# Over a remote connector this string is the model's whole briefing — the
# repo's skill file stays behind on the local side — so it has to stand alone
# rather than point at one.
_DEFAULT_INSTRUCTIONS = (
    "osaa-metrics composing server. Workflow: open_discovery → "
    "save_core_schema (widget) → build_core_table → build_core_summary → "
    "define_semantic_model → query_and_chart. "
    "The widget's Save reply goes to the widget, not to you: "
    "when the user says they saved, or asks for the next step, "
    "call build_core_table; its error says if no core table "
    "schema is saved yet. "
    "query_and_chart requires a label argument and errors without one; "
    "always pass back the model_name that define_semantic_model returns."
)


class OSAAMetricsServer(FastMCP):
    """Composing FastMCP server for osaa-metrics.

    Constructor DI: ``data_source``, ``encoder_provider``, and ``models`` are
    explicit arguments; nothing is reached through a module-level singleton.

    Mounts both sub-servers at the default ``namespace=None``, so their tools
    keep flat names:

    - ``DiscoveryMCPServer(data_source, encoder_provider, models)`` — ``name="osaa-discovery"``
    - ``MCPSemanticModel(models=self.models)``                     — ``name="BSL"``
    """

    def __init__(
        self,
        data_source: DataSourceProvider,
        encoder_provider: EncoderProvider,
        models: dict[str, Any] | None = None,
        name: str = "osaa-metrics",
        instructions: str = _DEFAULT_INSTRUCTIONS,
        **kwargs: Any,
    ):
        super().__init__(name=name, instructions=instructions, **kwargs)
        self.data_source = data_source
        self.encoder_provider = encoder_provider
        self.models = models if models is not None else {}
        self._discovery = DiscoveryMCPServer(
            data_source=data_source,
            encoder_provider=encoder_provider,
            models=self.models,
        )
        self._bsl_mcp = MCPSemanticModel(models=self.models, name="BSL")
        self.mount(self._discovery)
        self.mount(self._bsl_mcp)
        self._register_tools()
        self._register_resources()
        self._register_routes()

    @property
    def con(self) -> ibis.BaseBackend:
        """Lazy backend — raises a named ToolValidationError inside tool calls
        when the data source is unreachable (graceful boot)."""
        return self.data_source.get_con()

    def _register_tools(self) -> None:
        @self.tool(
            name="build_core_table",
            description=(
                "Run the PIVOT against the master table using the saved core "
                "table schema. Materializes the wide core table in the runtime "
                "cache. Call when the user says they saved, or asks for the next "
                "step, before define_semantic_model."
            ),
        )
        async def build_core_table(ctx: Context) -> dict:
            catalogue_df = self.data_source.get_catalogue()
            return await handle_build_core_table(
                ctx, self.con, catalogue_df, self.models
            )

        @self.tool(
            name="define_semantic_model",
            description=(
                "Validate a BSL semantic-model YAML against the materialized "
                "core table and register the resulting core semantic table "
                "under its YAML-declared name. On success pass the returned "
                "model_name to query_and_chart."
            ),
        )
        async def define_semantic_model(
            ctx: Context,
            yaml_text: Annotated[
                str, Field(description="YAML validated against bsl_model.schema.json")
            ],
        ) -> dict:
            return await handle_define_semantic_model(ctx, yaml_text, self.models)

        @self.tool(
            name="build_core_summary",
            description=(
                "Compute column-level statistics over the materialized core "
                "table (core_df). Read this BEFORE authoring the YAML — it "
                "tells you which columns are numeric, what ranges, where "
                "nulls cluster."
            ),
            app=AppConfig(resource_uri=_CORE_SUMMARY_URI),
        )
        async def build_core_summary(ctx: Context) -> dict:
            return await handle_build_core_summary(ctx)

        @self.tool(
            name="reset_state",
            description=(
                "Clear the recipe, the runtime cache, and this session's "
                "core semantic tables."
            ),
        )
        async def reset_state(
            ctx: Context,
            reason: Annotated[
                str,
                Field(
                    default="user requested",
                    description="Why state is being cleared — recorded for audit",
                ),
            ] = "user requested",
        ) -> dict:
            return await handle_reset_state(ctx, reason, models=self.models)

        @self.tool(
            name="query_and_chart",
            description=(
                "Compose a BSL query against a registered core semantic table "
                "and render the result as a vega-lite chart in the chart widget. "
                "ALWAYS requires chart_spec. chart_spec accepts a flat "
                "vega-lite dict or BSL's {backend, spec, format}; the grammar is "
                "identical to the query_model tool's chart_spec parameter."
            ),
            app=AppConfig(resource_uri=_CHART_URI),
        )
        async def query_and_chart(
            ctx: Context,
            model_name: Annotated[
                str, Field(description="Name of the registered core semantic table")
            ],
            chart_spec: Annotated[
                dict,
                BeforeValidator(_parse_json_string),
                Field(
                    description=(
                        "Vega-lite spec dict OR BSL wrapper form "
                        "{backend, spec, format}. Backend/format are accepted "
                        "for query_model symmetry but the iframe always "
                        "renders altair+json."
                    ),
                ),
            ],
            label: Annotated[
                str,
                Field(
                    description=(
                        "Short label for this query/chart pair (a few words), "
                        "distinct from and shorter than the chart title. Used to "
                        "list saved charts; stored in session.json."
                    ),
                ),
            ],
            dimensions: Annotated[
                list[str] | None,
                BeforeValidator(_parse_json_string),
                Field(
                    default=None,
                    description=(
                        "List of dimension field names. ONE name per entry — "
                        "e.g. ['region', 'year'], not ['region, year']."
                    ),
                ),
            ] = None,
            measures: Annotated[
                list[str] | None,
                BeforeValidator(_parse_json_string),
                Field(
                    default=None,
                    description=(
                        "List of measure field names. ONE name per entry — "
                        "e.g. ['avg_growth', 'total_gdp'], not "
                        "['avg_growth, total_gdp']."
                    ),
                ),
            ] = None,
            filters: Annotated[
                list[dict[str, Any]] | None,
                BeforeValidator(_parse_json_string),
                Field(default=None, description="BSL filter specs (JSON dicts)"),
            ] = None,
            order_by: Annotated[
                list | None,
                BeforeValidator(_parse_json_string),
                Field(
                    default=None, description="Order-by tuples like [['year','asc']]"
                ),
            ] = None,
            limit: Annotated[
                int | None,
                Field(default=None, description="Row limit"),
            ] = None,
        ) -> dict:
            return await handle_query_and_chart(
                ctx,
                self.models,
                model_name,
                dimensions or [],
                measures or [],
                filters,
                order_by,
                limit,
                chart_spec,
                label,
            )

        @self.tool(
            name="get_bsl_schema",
            description=(
                "Return the BSL model JSON Schema (bsl_model.schema.json) as a "
                "dict. Use this to author YAML for define_semantic_model. "
                "Mirrored as a resource at bsl-schema://osaa-metrics/v1."
            ),
        )
        async def get_bsl_schema(ctx: Context) -> dict:
            return handle_get_bsl_schema()

        @self.tool(
            name="get_session_schema",
            description=(
                "Return the session JSON Schema (session.schema.json) as a "
                "dict. Use this to validate a session.json before calling "
                "load_session. Mirrored as a resource at "
                "session-schema://osaa-metrics/v1."
            ),
        )
        async def get_session_schema(ctx: Context) -> dict:
            return handle_get_session_schema()

        @self.tool(
            name="load_session",
            description=(
                "Bootstrap state from a downloaded session.json. Validates the "
                "input; on success replaces the current recipe and runtime "
                "cache (validate-then-write ordering). Raises an error on "
                "validation failure; the message either carries errors as a "
                "JSON object, or is a plain-text refusal with no JSON body "
                "(e.g. a YAML-without-schema payload, or a field that is too "
                "long or contains disallowed characters); a YAML failure "
                "after a valid schema still leaves the core table already "
                "replaced, clearing only this session's own handle and "
                "model registration."
            ),
        )
        async def load_session(
            ctx: Context,
            json_text: Annotated[str, Field(description="Raw session.json content")],
        ) -> dict:
            catalogue_df = self.data_source.get_catalogue()
            return await handle_load_session(
                ctx, json_text, self.con, catalogue_df, self.models
            )

        @self.tool(
            name="download_session",
            description=(
                "Serialize current session state to session.json bytes, "
                "returned inline for the agent to hand to the user. Output "
                "carries schema_version='1', generated_at, and stable "
                "q-NNN ids per query."
            ),
        )
        async def download_session(ctx: Context) -> dict:
            return await handle_download_session(ctx)

        @self.tool(
            name="prepare_download",
            description=(
                "Generate a downloadable session.json (kind='session') or "
                "core-model YAML (kind='yaml') and return a one-time download "
                "URL. HTTP transport only: on stdio there is no download "
                "route and the URL will not work — use download_session "
                "instead. Links die on host restart; hand them to the user "
                "for immediate use."
            ),
        )
        async def prepare_download(ctx: Context, kind: str = "session") -> dict:
            base = self.data_source.settings.public_base_url
            return await handle_prepare_download(ctx, kind, base)

    def _register_resources(self) -> None:
        @self.resource(
            _CORE_SUMMARY_URI,
            app=AppConfig(csp=ResourceCSP(resource_domains=["https://unpkg.com"])),
        )
        def core_summary_view() -> str:
            return load_widget("core_summary.html")

        @self.resource(
            _CHART_URI,
            app=AppConfig(
                csp=ResourceCSP(
                    resource_domains=["https://unpkg.com", "https://cdn.jsdelivr.net"]
                )
            ),
        )
        def chart_view() -> str:
            return load_widget("chart.html")

        @self.resource(_BSL_SCHEMA_URI, mime_type="application/json")
        def bsl_schema_resource() -> dict:
            return handle_get_bsl_schema()

        @self.resource(_SESSION_SCHEMA_URI, mime_type="application/json")
        def session_schema_resource() -> dict:
            return handle_get_session_schema()

    def _register_routes(self) -> None:
        @self.custom_route("/health", methods=["GET"])
        async def health(request):
            return PlainTextResponse("ok")

        @self.custom_route("/download/{token}", methods=["GET"])
        async def download(request):
            from osaa_metrics.mcp import tools

            blob = tools._DOWNLOAD_TOKENS.pop(request.path_params["token"], None)  # noqa: SLF001  # module-private to this package, read by its own download route
            if blob is None:
                return PlainTextResponse("not found or expired", status_code=404)
            return Response(
                blob["bytes"],
                media_type=blob["mime_type"],
                headers={
                    "Content-Disposition": f'attachment; filename="{blob["filename"]}"',
                    # The bytes are agent-authored session JSON or YAML, so stop
                    # a host from MIME-sniffing them into something executable.
                    "X-Content-Type-Options": "nosniff",
                },
            )
