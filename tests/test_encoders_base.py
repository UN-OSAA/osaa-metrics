"""The Encoder Protocol is the public contract for query-time embedding.
This test pins the shape: same arg names, same kwargs, structural typing
so duck-typed implementations (e.g. SentenceTransformer) keep satisfying it."""

from __future__ import annotations

import inspect

from osaa_metrics.encoders.base import Encoder


def test_encoder_protocol_signature():
    """`encode` accepts `sentences` positionally and a `normalize_embeddings`
    keyword-only bool. Substrate callers rely on this exact shape; widening
    or renaming would force a coordinated change in discovery.py."""
    sig = inspect.signature(Encoder.encode)
    params = list(sig.parameters.values())
    # self, sentences, normalize_embeddings
    assert [p.name for p in params] == ["self", "sentences", "normalize_embeddings"]
    nb = sig.parameters["normalize_embeddings"]
    assert nb.kind is inspect.Parameter.KEYWORD_ONLY
    assert nb.default is True


def test_sentence_transformer_like_object_satisfies_protocol():
    """Structural typing — anything that quacks like SentenceTransformer.encode
    counts. Guards against accidental @runtime_checkable additions that would
    tighten the contract (Protocol must stay declarative for duck typing to
    work with third-party classes)."""

    class FakeST:
        def encode(self, sentences, *, normalize_embeddings: bool = True):
            return [[0.0] * 1024 for _ in sentences]

    enc: Encoder = FakeST()  # type: ignore[assignment]
    out = enc.encode(["hello"], normalize_embeddings=True)
    assert len(out) == 1
    assert len(out[0]) == 1024


def test_encoders_package_exports_the_protocol_and_the_local_encoder():
    from osaa_metrics import encoders

    assert encoders.__all__ == ["Encoder", "LocalEncoder"]
    assert not hasattr(encoders, "get_encoder")
