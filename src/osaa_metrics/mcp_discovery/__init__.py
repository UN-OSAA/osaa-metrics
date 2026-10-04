"""Discovery MCP sub-package — indicator search + working set commit.

BSL-free but not self-contained: the sub-package has zero
``boring_semantic_layer`` imports, while still importing from
``osaa_metrics.mcp.*`` — including the composing server's indicator search.
Those imports are what tie it to this package.
"""

from osaa_metrics.mcp_discovery.server import DiscoveryMCPServer

__all__ = ["DiscoveryMCPServer"]
