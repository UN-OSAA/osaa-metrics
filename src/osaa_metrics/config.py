"""Deployment settings — the single reader of the environment.

Deployment values enter the process here and only here, resolved into a
``Settings`` instance that feature code takes (or a value derived from
one) — the fields of ``Settings`` are the whole set.

Defaults resolve from the repo root computed package-relative
(``REPO_ROOT``), never from the working directory: launching the server from
any cwd yields the same configuration. A wheel install has no ``data/``
sample; connecting then fails with a named error telling the user to set
``OSAA_DATA_MASTER_URL`` / ``OSAA_DATA_META_URL``.

``.env`` at the repo root is loaded by ``load_settings`` itself (minimal
KEY=VALUE parser; real environment always wins), so the file is honoured for
any caller that imports the package, not just the ones whose runtime
happens to read it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ibis

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Resolved deployment values. Construct via :func:`load_settings`."""

    data_master_url: str
    data_meta_url: str
    bind_host: str
    port: int
    public_base_url: str


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, ``export`` prefix allowed,
    surrounding single/double quotes stripped, ``#`` lines and blanks ignored."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_settings(
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> Settings:
    """Resolve ``Settings`` from ``.env`` overlaid by the real environment.

    ``env`` defaults to ``os.environ``; pass a dict in tests. ``dotenv_path``
    defaults to ``REPO_ROOT / ".env"``.
    """
    if env is None:
        env = os.environ
    merged = _parse_dotenv(dotenv_path or REPO_ROOT / ".env")
    merged.update(env)

    port = int(merged.get("PORT") or "10000")
    return Settings(
        data_master_url=merged.get("OSAA_DATA_MASTER_URL")
        or str(REPO_ROOT / "data" / "master.parquet"),
        data_meta_url=merged.get("OSAA_DATA_META_URL")
        or str(REPO_ROOT / "data" / "meta.parquet"),
        bind_host=merged.get("OSAA_BIND_HOST") or "127.0.0.1",
        port=port,
        public_base_url=(
            merged.get("OSAA_PUBLIC_BASE_URL") or f"http://127.0.0.1:{port}"
        ).rstrip("/"),
    )


def build_connection(settings: Settings) -> ibis.BaseBackend:
    """Open an in-memory DuckDB ibis backend carrying the flat views
    (``master``, ``meta``) the substrate queries.

    The views read the parquet sources in place, so no data is copied into
    the process. Binding happens at view-creation time, so an unreachable
    source raises here; callers build the connection lazily and wrap that
    into a named tool error, which is what lets the server boot without a
    reachable source."""
    import ibis

    con = ibis.duckdb.connect()
    for name, source in (
        ("master", settings.data_master_url),
        ("meta", settings.data_meta_url),
    ):
        quoted = "'" + source.replace("'", "''") + "'"
        con.raw_sql(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({quoted})"  # noqa: S608  # name comes from the literal tuple above; the source URL is single-quote escaped
        )
    return con
