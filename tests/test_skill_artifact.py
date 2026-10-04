"""Verify the analyst playbook at skills/osaa-metrics/SKILL.md."""

from __future__ import annotations

from pathlib import Path

from osaa_metrics import __version__

SKILL = Path(__file__).resolve().parents[1] / "skills" / "osaa-metrics" / "SKILL.md"


def test_skill_has_frontmatter_and_bounded_sections() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---"), "missing YAML frontmatter"
    assert "name: osaa-metrics" in text
    assert "description:" in text
    # Two clearly-bounded sections so the discovery surface stays separable
    # from the composing workflow.
    assert "## Composing workflow" in text
    assert "## Discovery surface" in text


REFERENCE = SKILL.parent / "reference"


def test_skill_description_states_triggers_not_workflow():
    text = SKILL.read_text(encoding="utf-8")
    description = next(
        line for line in text.splitlines() if line.startswith("description:")
    )
    assert "Use when" in description
    # A description that lists the tool sequence gets followed instead of the body.
    for tool in ("open_discovery", "build_core_table", "query_and_chart", "→"):
        assert tool not in description, tool


def test_reference_files_exist_and_are_linked_one_level_deep():
    text = SKILL.read_text(encoding="utf-8")
    for name in ("charting.md", "sessions.md", "discovery.md"):
        assert (REFERENCE / name).is_file(), name
        assert f"reference/{name}" in text, f"SKILL.md does not point at {name}"
    # The always-loaded playbook stays short; roughly 300 of these words are
    # the vocabulary table, which is a hard rule and cannot move to a reference.
    body = SKILL.read_text(encoding="utf-8")
    assert len(body.split()) < 1100, "SKILL.md is meant to be the short playbook"


def test_discovery_reference_carries_the_separable_note():
    text = (REFERENCE / "discovery.md").read_text(encoding="utf-8")
    assert "separable — moves with `mcp-discovery`" in text


def test_charting_reference_teaches_title_subtitle_source():
    text = (REFERENCE / "charting.md").read_text(encoding="utf-8")
    for needle in ('"text"', '"subtitle"', "Source:", "80", "120"):
        assert needle in text, needle


def test_skill_teaches_canonical_vocabulary() -> None:
    text = SKILL.read_text(encoding="utf-8")
    for term in [
        "core table",
        "core semantic table",
        "Save",
        "working set",
        "discovery widget",
    ]:
        assert term in text, f"canonical term missing from skill: {term}"
    # The skill must instruct against inventing terminology.
    assert (
        "invent" in text.lower()
        or "do not coin" in text.lower()
        or "don't coin" in text.lower()
    )


def test_skill_is_written_for_the_class_not_the_instance():
    text = SKILL.read_text(encoding="utf-8")
    for instance_word in ("World Bank", "UNESCO", "OSAA economist"):
        assert instance_word not in text, instance_word


def test_skill_states_the_server_version_it_was_written_against():
    assert f"osaa-metrics {__version__}" in SKILL.read_text(encoding="utf-8")


def test_skill_carries_the_local_state_rules_and_a_walkthrough():
    text = SKILL.read_text(encoding="utf-8")
    assert "### State and sessions" in text
    assert "## Getting started" in text
    state = text.split("### State and sessions", 1)[1].split("###", 1)[0]
    assert "download_session" in state


def test_skill_state_section_does_not_leak_the_session_cap_count():
    text = SKILL.read_text(encoding="utf-8")
    state = text.split("### State and sessions", 1)[1].split("###", 1)[0]
    assert "16" not in state


SETUP_SKILL = SKILL.parent.parent / "osaa-setup" / "SKILL.md"


def test_setup_skill_is_a_runbook_over_the_recipes():
    text = SETUP_SKILL.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "name: osaa-setup" in text
    for recipe in ("just setup", "just data", "just check", "just encoder"):
        assert recipe in text, recipe
    assert "--frozen" in text and "--no-sync" in text
