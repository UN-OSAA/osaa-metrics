"""Smoke test: each module on the package's public surface imports cleanly."""

import importlib

PUBLIC_MODULES = [
    "osaa_metrics",
    "osaa_metrics.discovery",
    "osaa_metrics.core_table",
    "osaa_metrics.semantic",
    "osaa_metrics.session",
    "osaa_metrics.summarize",
    "osaa_metrics.theme",
    "osaa_metrics.mcp",
    "osaa_metrics.mcp.providers",
    "osaa_metrics.mcp_discovery",
]


def test_every_module_imports():
    """Importing the public surface should not raise."""
    for name in PUBLIC_MODULES:
        importlib.import_module(name)
