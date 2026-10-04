"""The curated public surface — what `from osaa_metrics import ...` guarantees."""


def test_public_surface_importable():
    import osaa_metrics as om

    expected = [
        "__version__",
        # config
        "Settings",
        "load_settings",
        "build_connection",
        # discovery
        "SearchResult",
        "load_embedding_cache",
        "search_indicators",
        "search_indicators_with_fallback",
        # core table
        "assign_var_names",
        "build_wide_table",
        "enrich_core_table",
        "load_from_csv",
        "slugify_indicator_name",
        # semantic
        "ModelValidationError",
        "validate_and_load_model",
        "summarize_model",
        "display_label",
        # summary + session
        "build_core_summary_payload_from_handle",
        "Session",
        "SessionValidationError",
        "dump_session",
        "load_session",
        "validate_session",
        # chart theme
        "register_theme",
    ]
    missing = [name for name in expected if not hasattr(om, name)]
    assert missing == []
    assert sorted(om.__all__) == sorted(expected)


def test_base_import_does_not_require_mcp_extra():
    """Importing the base package must not pull in the MCP sub-package.

    Checked in a fresh interpreter so the result reflects what ``import
    osaa_metrics`` actually does, unaffected by other tests (or this suite's
    conftest) having already imported ``osaa_metrics.mcp`` in-process.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import osaa_metrics, sys; "
            "loaded = sorted(m for m in sys.modules "
            "if m == 'osaa_metrics.mcp' or m.startswith('osaa_metrics.mcp.')); "
            "assert not loaded, loaded",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
