"""Encoder provider — process-scoped encoder holder.

The factory is ``LocalEncoder`` for the public server (see
``build_server``'s ``encoder_factory`` seam for how another backend is
injected). Tests inject a fake factory directly to avoid loading real model
weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

from osaa_metrics.encoders import Encoder

if TYPE_CHECKING:
    from collections.abc import Callable


class EncoderProvider:
    """Server-scoped encoder, built once.

    One instance per MCP process; the factory runs on the first ``get()``
    call and the encoder it returns is cached for later calls. A factory
    that raises leaves nothing cached, so the next ``get()`` runs it again;
    with the model files pinned and the network forbidden that retry never
    downloads. A factory that returns ``None`` is treated as a failure too."""

    def __init__(self, factory: Callable[[], Encoder]) -> None:
        self._factory = factory
        self._encoder: Encoder | None = None

    def get(self) -> Encoder:
        """Return the encoder: the factory builds it on the first call, and
        later calls return the same one. Raises ``RuntimeError`` if the
        factory returns ``None``."""
        if self._encoder is None:
            encoder = self._factory()
            if encoder is None:
                raise RuntimeError(
                    "encoder factory returned None; an Encoder is required"
                )
            self._encoder = encoder
        return self._encoder

    def reset(self) -> None:
        """Drop the cached encoder; primarily for tests."""
        self._encoder = None
