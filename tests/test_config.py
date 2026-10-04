"""Settings module — single reader of the environment."""

from dataclasses import FrozenInstanceError, fields

import duckdb
import pytest

from osaa_metrics.config import REPO_ROOT, Settings, build_connection, load_settings


def test_defaults_resolve_package_relative_not_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # cwd must not matter
    for var in (
        "OSAA_DATA_MASTER_URL",
        "OSAA_DATA_META_URL",
        "OSAA_BIND_HOST",
        "PORT",
        "OSAA_PUBLIC_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    s = load_settings(env={}, dotenv_path=tmp_path / ".env")
    assert s.data_master_url == str(REPO_ROOT / "data" / "master.parquet")
    assert s.data_meta_url == str(REPO_ROOT / "data" / "meta.parquet")
    assert s.bind_host == "127.0.0.1"
    assert s.port == 10000
    assert s.public_base_url == "http://127.0.0.1:10000"


def test_env_overrides_every_field():
    s = load_settings(
        env={
            "OSAA_DATA_MASTER_URL": "https://example.org/m.parquet",
            "OSAA_DATA_META_URL": "https://example.org/x.parquet",
            "OSAA_BIND_HOST": "0.0.0.0",
            "PORT": "9000",
            "OSAA_PUBLIC_BASE_URL": "https://svc.example.org",
        }
    )
    assert s.data_master_url == "https://example.org/m.parquet"
    assert s.data_meta_url == "https://example.org/x.parquet"
    assert s.bind_host == "0.0.0.0"
    assert s.port == 9000
    assert s.public_base_url == "https://svc.example.org"


def test_settings_has_no_encoder_choice():
    """The encoder is not a deployment value: the public server builds its
    encoder by construction, not from configuration. Settings holds exactly
    these fields, so a new one fails here and gets a deliberate look."""
    s = load_settings(env={})
    assert not hasattr(s, "encoder")
    assert {f.name for f in fields(Settings)} == {
        "data_master_url",
        "data_meta_url",
        "bind_host",
        "port",
        "public_base_url",
    }


def test_dotenv_loaded_but_real_env_wins(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment line\n"
        "export OSAA_DATA_MASTER_URL=/from/dotenv/master.parquet\n"
        'OSAA_DATA_META_URL="/from/dotenv/meta.parquet"\n'
        "PORT=7777\n"
    )
    s = load_settings(env={"PORT": "8888"}, dotenv_path=dotenv)
    assert s.data_master_url == "/from/dotenv/master.parquet"  # from .env
    assert s.data_meta_url == "/from/dotenv/meta.parquet"  # quotes stripped
    assert s.port == 8888  # real env wins


def test_missing_dotenv_is_fine(tmp_path):
    s = load_settings(env={}, dotenv_path=tmp_path / "absent.env")
    assert isinstance(s, Settings)


def test_settings_is_frozen():
    s = load_settings(env={})
    with pytest.raises(FrozenInstanceError):
        s.port = 1  # type: ignore[misc]


@pytest.mark.needs_data
def test_build_connection_registers_flat_master_and_meta():
    s = load_settings()  # conftest routes OSAA_DATA_* at the local mirror
    con = build_connection(s)
    tables = set(con.list_tables())
    assert {"master", "meta"} <= tables
    assert con.table("master").count().execute() > 0


def test_build_connection_bad_source_raises(tmp_path):
    s = load_settings(
        env={
            "OSAA_DATA_MASTER_URL": str(tmp_path / "absent.parquet"),
            "OSAA_DATA_META_URL": str(tmp_path / "absent.parquet"),
        }
    )
    with pytest.raises(duckdb.IOException, match="No files found"):
        build_connection(s)


def test_build_connection_quotes_single_quotes(tmp_path):
    # A source path containing a quote must not break the SQL string.
    s = load_settings(
        env={
            "OSAA_DATA_MASTER_URL": str(tmp_path / "o'brien.parquet"),
            "OSAA_DATA_META_URL": str(tmp_path / "o'brien.parquet"),
        }
    )
    with pytest.raises(duckdb.IOException, match="No files found") as exc_info:
        build_connection(s)
    assert (
        "syntax" not in str(exc_info.value).lower()
    )  # missing-file error, not SQL breakage
