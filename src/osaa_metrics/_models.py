"""Model kernel — the pinned BGE-M3 files and the operations on them.

``download_model`` puts one revision of the hub repo into the hub cache
(the library default location, ``~/.cache/huggingface/hub`` unless the hub
library's own environment says otherwise). ``load_model`` reads that same
revision from the cache into RAM and reads only the cache
(``local_files_only``), so a missing file raises instead of downloading; the
server's entry points forbid the network altogether with ``HF_HUB_OFFLINE``.
The pin is a commit hash, so the cache never grows a second snapshot because
the repo's ``main`` moved.

The private leading underscore signals: this is a leaf module
``encoders.local`` depends on; callers do not reach back through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from sentence_transformers import SentenceTransformer

MODEL_REPO = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
# The files a SentenceTransformer load of this repo reads. The repo also
# carries an ONNX export, images, and auxiliary heads the encoder never
# uses; leaving them off the list keeps the download to the weights and the
# tokenizer.
MODEL_FILES: tuple[str, ...] = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/*",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "pytorch_model.bin",
)


def _snapshot(*, local_files_only: bool, downloader: Callable[..., str] | None) -> str:
    if downloader is None:
        # Imported here so importing this module does not load huggingface_hub.
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    kwargs = {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "allow_patterns": list(MODEL_FILES),
    }
    if local_files_only:
        kwargs["local_files_only"] = True
    return downloader(**kwargs)


def download_model(*, downloader: Callable[..., str] | None = None) -> str:
    """Download the pinned model files into the hub cache; return the snapshot path.

    A hub operation only: the model runtime is not imported, so ``just
    encoder`` runs it without paying for torch. Files already in the cache
    are not downloaded again.
    ``downloader`` is the test seam; the default is the hub library's
    ``snapshot_download``.
    """
    return _snapshot(local_files_only=False, downloader=downloader)


def cached_model_path(*, downloader: Callable[..., str] | None = None) -> str:
    """Return the cached snapshot path for the pinned revision without any
    network call; raise when the cache's tree listing for the pinned
    revision shows a pinned file missing.

    ``just check`` uses this to tell "run ``just encoder``" from "ready"
    without a network call.
    """
    return _snapshot(local_files_only=True, downloader=downloader)


def load_model(model_name: str = MODEL_REPO) -> SentenceTransformer:
    """Load the pinned revision from the hub cache into RAM.

    ``local_files_only`` makes the load read only the cache: a cache that
    ``download_model`` has not filled raises here at once, and the caller's
    failure path takes over. The import is inside the function so importing
    this module costs nothing; the model runtime loads only when a model is
    actually loaded.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name, revision=MODEL_REVISION, local_files_only=True
    )
