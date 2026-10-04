# osaa-metrics — the install story. Run these from the repository root.
# Needs uv (https://docs.astral.sh/uv/) and just (https://github.com/casey/just).

set shell := ["bash", "-euo", "pipefail", "-c"]

sample_base := "https://huggingface.co/datasets/spencerlima/osaa-metrics/resolve/main"

# List the recipes
default:
    @just --list

# Install the server and the encoder runtime into .venv
setup:
    @command -v uv >/dev/null || { echo "uv is not installed — see https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
    uv sync --frozen --extra mcp --extra encoder

# Download the sample data (two parquet files) into data/
data:
    @mkdir -p data
    @for f in master.parquet meta.parquet; do \
        if [ -s "data/$f" ]; then \
            echo "data/$f present — skipping"; \
        else \
            curl -fL --progress-bar -o "data/$f.part" "{{sample_base}}/$f" \
                && mv "data/$f.part" "data/$f"; \
        fi; \
    done

# Download the pinned BGE-M3 model files (about 2.3 GB) into the model cache; prints where they landed
encoder:
    uv run --frozen python -c "from osaa_metrics._models import download_model; print(download_model())"

# Report whether the data files open and the model files are cached
check:
    #!/usr/bin/env -S uv run --frozen python
    import sys
    from pathlib import Path

    from osaa_metrics.config import REPO_ROOT, build_connection, load_settings

    settings = load_settings()
    failed = False

    def show(source: str) -> str:
        if "://" in source:
            return source
        p = Path(source)
        if p.is_absolute() and p.is_relative_to(REPO_ROOT):
            return str(p.relative_to(REPO_ROOT))
        return source

    for env_var, source in (
        ("OSAA_DATA_MASTER_URL", settings.data_master_url),
        ("OSAA_DATA_META_URL", settings.data_meta_url),
    ):
        if "://" not in source and not Path(source).is_file():
            print(
                f"data:    {show(source)} missing — run `just data` for the default "
                f"sample path, or point {env_var} at an existing file in .env"
            )
            failed = True

    if not failed:
        try:
            con = build_connection(settings)
            indicators = con.table("meta").count().execute()
        except Exception as exc:
            print(f"data:    cannot open the data sources — {exc}")
            print("         delete a local file that will not open and run `just data`; a URL source is changed in .env")
            failed = True
        else:
            print(
                f"data:    {show(settings.data_master_url)} OK · "
                f"{show(settings.data_meta_url)} OK ({indicators} indicators)"
            )

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("encoder: sentence-transformers is not installed — run `just setup`")
        failed = True
    except Exception as exc:
        print(f"encoder: sentence-transformers failed to import — {exc}")
        failed = True
    else:
        from huggingface_hub.constants import HF_HUB_CACHE

        from osaa_metrics._models import MODEL_REPO, cached_model_path

        try:
            cached_model_path()
        except Exception as exc:
            print(
                f"encoder: {MODEL_REPO} is not in the model cache at {HF_HUB_CACHE} "
                f"— run `just encoder` ({exc})"
            )
            failed = True
        else:
            print(f"encoder: {MODEL_REPO} cached at {HF_HUB_CACHE} — semantic search enabled")

    sys.exit(1 if failed else 0)

# Open the playground: rebuild a chat session's core table, queries and charts in the browser
app: data
    uv run --frozen --extra app marimo run app/playground.py

# Open a worked example notebook from examples/ in marimo, in its own sandbox built from the notebook's header
example name:
    uv run --with "marimo>=0.23" marimo edit --sandbox examples/{{name}}/{{name}}.py
