"""LocalEncoder — in-process BGE-M3 via sentence-transformers.

The encoder the public server runs. The model files come from the cache
that ``just encoder`` filled; the load reads only the cache, so a missing
file raises rather than downloads. The
weights stay in RAM for the life of the process; every query is then local
and offline.

The heavy work — instantiating
:class:`~sentence_transformers.SentenceTransformer` via :func:`load_model` —
happens when the encoder is constructed; the server constructs it once at
start, so no query pays for it.

The factory indirection (``factory=load_model``) is the test seam — pass
a fake to unit-test without loading the real weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

from osaa_metrics._models import load_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from osaa_metrics.encoders.base import Encoder


class LocalEncoder:
    """In-process encoder; the model loads when the encoder is constructed. Satisfies :class:`osaa_metrics.encoders.base.Encoder` structurally."""

    def __init__(self, *, factory: Callable[[], Encoder] = load_model) -> None:
        self._model = factory()

    def encode(self, sentences, *, normalize_embeddings: bool = True):
        """Encode sentences with the loaded model.

        Args:
            sentences: Iterable of strings to embed.
            normalize_embeddings: If True (default), return unit-length vectors.

        Returns:
            A container indexable into 1024-element float vectors (numpy
            array, per :class:`sentence_transformers.SentenceTransformer`).
        """
        return self._model.encode(sentences, normalize_embeddings=normalize_embeddings)
