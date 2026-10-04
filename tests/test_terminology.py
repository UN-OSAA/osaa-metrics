"""Public text never says "semantic search not enabled": the discovery
tool has one note, "ranking unavailable", and no file a user reads may
carry the other phrase."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PUBLIC_TEXT = ("src", "skills", "README.md", "AGENTS.md", "justfile")


def _public_files():
    for entry in PUBLIC_TEXT:
        path = REPO / entry
        if path.is_file():
            yield path
        else:
            # Skip bytecode: a .pyc carries the string constants of whatever source
            # compiled it and is not public text.
            yield from (
                p
                for p in path.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            )


def test_the_retired_note_is_gone_from_public_text():
    hits = [
        str(p.relative_to(REPO))
        for p in _public_files()
        if "semantic search not enabled"
        in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert hits == [], hits
