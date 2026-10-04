"""The Encoder Protocol — the contract every encoder backend must satisfy.

Substrate callers in :mod:`osaa_metrics.discovery` invoke
``encoder.encode([query], normalize_embeddings=True)[0]`` — a sequence in,
a sequence of 1024-dim vectors out. Single-query convenience is the
caller's responsibility; this keeps the contract aligned with
``sentence_transformers.SentenceTransformer.encode`` so the in-process
case is a zero-wrapper drop-in."""

from __future__ import annotations

from typing import Protocol


class Encoder(Protocol):
    """Minimal duck-typed encoder contract.

    Implementations must accept any iterable of strings and return a
    container that's indexable into 1024-element float vectors (lists, tuples,
    or numpy arrays — substrate normalises the row via ``[float(x) for x in qvec]``
    inside :func:`osaa_metrics.discovery.search_indicators`).
    """

    def encode(self, sentences, *, normalize_embeddings: bool = True):
        """Embed each sentence; return a container indexable into one
        1024-element float vector per sentence."""
