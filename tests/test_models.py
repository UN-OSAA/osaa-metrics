"""The model kernel pins one revision of one hub repo and never downloads at
load time. Tests inject fakes: the suite must pass without the encoder
extra installed and without network."""

from __future__ import annotations

import sys
import types

import pytest

from osaa_metrics import _models


def test_pin_is_a_full_commit_hash():
    assert _models.MODEL_REPO == "BAAI/bge-m3"
    assert len(_models.MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in _models.MODEL_REVISION)


def test_file_list_names_the_weights_and_excludes_the_extras():
    files = set(_models.MODEL_FILES)
    assert "pytorch_model.bin" in files
    assert "modules.json" in files
    assert not any(f.startswith(("onnx", "imgs")) for f in files)


def test_download_model_pins_revision_and_file_list():
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/snap"

    assert _models.download_model(downloader=fake_download) == "/snap"
    (kwargs,) = calls
    assert kwargs == {
        "repo_id": _models.MODEL_REPO,
        "revision": _models.MODEL_REVISION,
        "allow_patterns": list(_models.MODEL_FILES),
    }


def test_cached_model_path_never_touches_the_network():
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/snap"

    assert _models.cached_model_path(downloader=fake_download) == "/snap"
    (kwargs,) = calls
    assert kwargs["local_files_only"] is True
    assert kwargs["revision"] == _models.MODEL_REVISION


def test_download_model_does_not_need_the_encoder_extra(monkeypatch):
    """Downloading is a hub operation; importing the model runtime for it
    would drag torch into `just encoder`."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _models.download_model(downloader=lambda **kw: "/snap") == "/snap"


def test_load_model_loads_the_pinned_revision_offline(monkeypatch):
    recorded = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            recorded["name"] = name
            recorded.update(kwargs)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    _models.load_model()

    assert recorded == {
        "name": _models.MODEL_REPO,
        "revision": _models.MODEL_REVISION,
        "local_files_only": True,
    }


def test_load_model_rejects_a_missing_cache_instead_of_downloading(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    with pytest.raises(OSError, match="not cached"):
        _models.load_model()
