"""The install story: the justfile at the repo root, README.md, and
skills/osaa-setup/SKILL.md. Holds those three surfaces to one
canonical install sync (`uv sync --frozen --extra mcp --extra encoder`), and
checks the justfile's recipes for the shapes the install path depends on
— recipe names, the sample-data fetch, the encoder extra, and `check`'s
failure-naming text."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JUSTFILE = REPO / "justfile"

CANONICAL_SYNC = "uv sync --frozen --extra mcp --extra encoder"
SAMPLE_BASE = "https://huggingface.co/datasets/spencerlima/osaa-metrics/resolve/main"


def _recipe_body(name: str) -> str:
    """A top-level recipe's own body: every indented line following its
    `name:` header, up to (not including) the next top-level line."""
    lines = JUSTFILE.read_text(encoding="utf-8").splitlines()
    body: list[str] = []
    in_recipe = False
    for line in lines:
        if line.startswith(f"{name}:"):
            in_recipe = True
            continue
        if in_recipe:
            if line and not line[0].isspace():
                break
            body.append(line)
    return "\n".join(body)


def _recipe_names() -> set[str]:
    """Top-level recipe names: a line that starts at column 0 and is not a
    comment, split at its first colon into a head and a rest. The head's
    first word is the recipe name (any further words are its parameters),
    unless the colon is a `name := value` variable declaration (rest
    starting with `=`)."""
    names = set()
    for line in JUSTFILE.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and ":" in line and not line.startswith("#"):
            head, _, rest = line.partition(":")
            head = head.strip()
            if head and not rest.strip().startswith("="):
                names.add(head.split()[0])
    return names


def test_justfile_recipes():
    assert {"setup", "data"} <= _recipe_names()


def test_recipe_names_excludes_variable_declarations():
    assert "sample_base" not in _recipe_names()


def test_data_fetches_both_sample_files_from_the_published_dataset():
    text = JUSTFILE.read_text(encoding="utf-8")
    assert SAMPLE_BASE in text
    assert "master.parquet" in text and "meta.parquet" in text


def test_data_dir_is_a_tracked_empty_target():
    assert (REPO / "data" / ".gitkeep").is_file()


def test_encoder_and_check_recipes_exist():
    assert {"encoder", "check"} <= _recipe_names()


def test_encoder_recipe_only_downloads_the_model_files():
    """`encoder` is the download step and nothing else: no sync (setup owns
    that) and no model load (the server owns that)."""
    body = _recipe_body("encoder")
    assert "download_model" in body
    assert "uv sync" not in body
    assert "load_model" not in body.replace("download_model", "")


def test_setup_installs_the_encoder_extra():
    assert CANONICAL_SYNC in _recipe_body("setup")


@pytest.mark.skipif(shutil.which("just") is None, reason="just is not installed")
def test_just_itself_parses_the_recipe_set():
    """The hand parser above re-implements just's recipe grammar; this test
    asks just itself, so a bad `set` directive or an unbalanced `{{ }}`
    interpolation fails here even when it would slip past text-level
    assertions. `--summary` only parses the justfile's grammar, though — a
    shebang recipe's body is opaque text to it, so an invalid script inside
    one still exits 0."""
    result = subprocess.run(
        ["just", "--summary"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert set(result.stdout.split()) == _recipe_names()


def test_check_names_a_fix_when_the_data_sources_cannot_be_opened():
    text = JUSTFILE.read_text(encoding="utf-8")
    assert "delete a local file that will not open and run `just data`" in text
    assert "a URL source is changed in .env" in text


@pytest.mark.skipif(shutil.which("just") is None, reason="just is not installed")
def test_check_reports_a_local_encoder_runtime_failure_instead_of_crashing(tmp_path):
    """A `local`-encoder failure that isn't a plain `ImportError` — a broken
    install raising mid-import, say — is printed and counted as a failure
    the way the recipe's other branches report theirs, not left to
    traceback out of `check`."""
    stub = tmp_path / "sentence_transformers.py"
    stub.write_text("raise RuntimeError('stub model runtime failure')\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        ["just", "check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "stub model runtime failure" in result.stdout


@pytest.mark.skipif(shutil.which("just") is None, reason="just is not installed")
def test_check_names_the_download_recipe_when_the_model_is_not_cached(tmp_path):
    """An installed runtime with an empty model cache is the state a user is
    in between `just setup` and `just encoder`; `check` must say which recipe
    fixes it and where it looked. Both libraries are stubbed on PYTHONPATH
    so the test needs neither the encoder extra nor a real cache."""
    (tmp_path / "sentence_transformers.py").write_text("")
    hub = tmp_path / "huggingface_hub"
    hub.mkdir()
    (hub / "__init__.py").write_text(
        "def snapshot_download(**kwargs):\n"
        "    raise FileNotFoundError('stub: pinned files are not cached')\n"
    )
    (hub / "constants.py").write_text('HF_HUB_CACHE = "/stub/hub-cache"\n')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        ["just", "check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "run `just encoder`" in result.stdout
    assert "/stub/hub-cache" in result.stdout


SETUP_SKILL = REPO / "skills" / "osaa-setup" / "SKILL.md"


def test_setup_skill_quotes_the_canonical_sync():
    assert CANONICAL_SYNC in SETUP_SKILL.read_text(encoding="utf-8")


README = REPO / "README.md"

# Strings the README must not contain. The reason differs per entry — an
# unused name, a stale figure, an internal upstream detail the public
# install story doesn't need, a rejected term — and being true elsewhere in
# the codebase is not a defense against appearing here.
DEAD_CLAIMS = (
    "profiles.yml",
    "osaa_r2",
    "BSL_PROFILE",
    "silver.master",
    "Cockpit",
    "1.3 GB",
    "uvx",
    "--extra local",
    "0.0.0.0",
    "r2.dev",
    "git@github.com",
    "Semantic search (optional)",
    "OSAA_METRICS_ENCODER",
    "keyword search only",
)


def test_readme_quotes_the_canonical_sync_and_the_recipes():
    text = README.read_text(encoding="utf-8")
    assert CANONICAL_SYNC in text
    for recipe in ("just setup", "just data", "just check", "just encoder"):
        assert recipe in text, recipe


def test_readme_carries_no_dead_claims():
    text = README.read_text(encoding="utf-8")
    present = [claim for claim in DEAD_CLAIMS if claim in text]
    assert not present, present


def test_readme_launcher_cannot_reshape_the_venv():
    text = README.read_text(encoding="utf-8")
    assert '"--frozen", "--no-sync"' in text
