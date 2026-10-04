"""Encoder sub-package — the query-time embedding backend.

Public surface:
- :class:`Encoder` Protocol (the contract)
- :class:`LocalEncoder` (the in-process BGE-M3 backend, the one the public
  server uses)

Another backend reaches the server only through ``build_server``'s
``encoder_factory`` seam."""

from __future__ import annotations

from osaa_metrics.encoders.base import Encoder
from osaa_metrics.encoders.local import LocalEncoder

__all__ = ["Encoder", "LocalEncoder"]
