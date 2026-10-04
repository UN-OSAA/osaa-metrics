"""LocalEncoder is the in-process BGE-M3 backend.

Tests inject a fake SentenceTransformer factory at construction time so the
suite never downloads the real weights — the load-on-construction and
pass-through contracts are what is under test, not the model itself."""

from __future__ import annotations

from osaa_metrics.encoders.local import LocalEncoder


class _FakeST:
    """Minimal stand-in: returns shape-correct fake vectors."""

    def __init__(self):
        self.calls = []

    def encode(self, sentences, *, normalize_embeddings: bool = True):
        self.calls.append((list(sentences), normalize_embeddings))
        return [[0.0] * 1024 for _ in sentences]


def test_local_encoder_loads_the_model_on_construction():
    """Construction runs the factory once; encode() never loads. Mirrors the
    server's load-at-start: EncoderProvider.get() constructs the encoder
    before the first request."""
    calls = []

    def factory():
        calls.append(1)
        return _FakeST()

    enc = LocalEncoder(factory=factory)
    assert calls == [1]
    enc.encode(["hello"], normalize_embeddings=True)
    assert calls == [1]


def test_local_encoder_reuses_loaded_model():
    """Subsequent .encode() calls must reuse the same SentenceTransformer
    instance — re-loading BGE-M3 per query would be a perf disaster."""
    calls = []

    def factory():
        calls.append(1)
        return _FakeST()

    enc = LocalEncoder(factory=factory)
    enc.encode(["a"], normalize_embeddings=True)
    enc.encode(["b"], normalize_embeddings=True)
    enc.encode(["c"], normalize_embeddings=True)
    assert calls == [1], "factory must run exactly once across multiple encodes"


def test_local_encoder_forwards_args_to_underlying_model():
    """encode() must pass through both `sentences` and `normalize_embeddings`
    verbatim — the substrate's `[float(x) for x in qvec]` assumes the
    underlying model returned a normalised vector when asked."""
    fake = _FakeST()
    enc = LocalEncoder(factory=lambda: fake)
    result = enc.encode(["x", "y"], normalize_embeddings=True)
    assert fake.calls == [(["x", "y"], True)]
    assert len(result) == 2
    assert len(result[0]) == 1024


def test_local_encoder_default_factory_is_load_model():
    """The default factory is `_models.load_model`, so `LocalEncoder()` with
    no args loads BGE-M3 in-process. Verified on the signature, not by
    construction — the real load is too expensive for the unit suite."""
    import inspect

    from osaa_metrics import _models

    assert (
        inspect.signature(LocalEncoder.__init__).parameters["factory"].default
        is _models.load_model
    )
