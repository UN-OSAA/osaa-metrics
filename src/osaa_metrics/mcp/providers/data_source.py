"""Data-source provider — lazy process-scoped DuckDB/ibis backend + gold-meta
catalogue.

The connection is built from :class:`osaa_metrics.config.Settings` on the
first ``get_con()`` call — never at server construction — so a missing or
unreachable source surfaces as a named tool error on the first tool call
instead of a pre-handshake process death. The ``con_factory`` argument is the
test seam, mirroring ``EncoderProvider(factory=...)``.

The catalogue (gold-meta metadata, no embeddings) is reference data that
``enrich_core_table`` uses to back-fill ``(source, database)`` from indicator
codes. Loaded once on the first ``get_catalogue()`` call and reused for the
lifetime of the MCP process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from osaa_metrics.config import Settings, build_connection
from osaa_metrics.discovery import load_embedding_cache
from osaa_metrics.mcp.tools import ToolValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    import ibis


class DataSourceProvider:
    """Process-scoped lazy connection + catalogue cache."""

    def __init__(
        self,
        settings: Settings,
        con_factory: Callable[[Settings], ibis.BaseBackend] = build_connection,
    ) -> None:
        self._settings = settings
        self._con_factory = con_factory
        self._con: ibis.BaseBackend | None = None
        self._catalogue_df: pl.DataFrame | None = None

    @property
    def settings(self) -> Settings:
        """The ``Settings`` this provider was built with."""
        return self._settings

    def get_con(self) -> ibis.BaseBackend:
        """Return the backend, building it on first call.

        Wraps any construction failure in ``ToolValidationError`` so the agent
        sees an actionable, named error instead of a raw traceback."""
        if self._con is None:
            try:
                self._con = self._con_factory(self._settings)
            except Exception as e:
                raise ToolValidationError(
                    f"data source unavailable: {e}. Set OSAA_DATA_MASTER_URL / "
                    f"OSAA_DATA_META_URL to readable parquet paths or URLs "
                    f"(current: master={self._settings.data_master_url!r}, "
                    f"meta={self._settings.data_meta_url!r})."
                ) from e
        return self._con

    def get_catalogue(self) -> pl.DataFrame:
        """Lazy-loaded gold-meta catalogue (no embeddings)."""
        if self._catalogue_df is None:
            self._catalogue_df = load_embedding_cache(con=self.get_con())
        return self._catalogue_df

    def reset(self) -> None:
        """Drop the cached connection and catalogue; primarily for tests."""
        self._con = None
        self._catalogue_df = None
