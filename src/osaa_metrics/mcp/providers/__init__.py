"""Provider classes for DataSource and Encoder.

``build_server`` constructs these and hands them to ``OSAAMetricsServer``,
which holds them as ``self.data_source`` and ``self.encoder_provider``
(constructor DI) and passes the same two objects down to the mounted
``DiscoveryMCPServer``; the backend itself is reached lazily through the
``self.con`` property. There are no module-level instances — that pattern is
incompatible with the constructor-DI MCP standard.
"""

from __future__ import annotations

from osaa_metrics.mcp.providers.data_source import DataSourceProvider
from osaa_metrics.mcp.providers.encoder import EncoderProvider

__all__ = [
    "DataSourceProvider",
    "EncoderProvider",
]
