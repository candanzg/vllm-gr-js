# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast detokenization helpers that bypass the HuggingFace Python wrapper.

``transformers`` ``batch_decode`` / batched ``decode`` first recursively
converts every token id through the pure-Python ``to_py_obj`` helper and then
loops over sequences in Python, calling the single-sequence Rust ``decode``
once per beam.  The underlying ``tokenizers.Tokenizer`` (``tokenizer._tokenizer``)
already exposes a natively batched ``decode_batch`` (and a single ``decode``)
implemented in Rust.  Routing straight to those methods removes the
``to_py_obj`` traversal and the per-sequence Python call frames.

Both ``decode`` and ``decode_batch`` on the Rust tokenizer are ``&self``
(PyO3) methods, so they are safe to call concurrently without the
``TokenizerPool`` deep-copy borrow.

Correctness guard (behavior probe): merely having a ``_tokenizer`` backend is
not proof that bypassing the HuggingFace wrapper is byte-safe.  Since
transformers v5 even legacy tokenizers (e.g. ``Qwen2Tokenizer``) subclass the
tokenizers backend and report ``is_fast=True``, an ``isinstance`` check cannot
distinguish them.  Such a legacy tokenizer's wrapper still augments
added-token decoding: special ids such as ``<|im_start|>`` decode to their
marker text through the wrapper but to an **empty string** through the raw
Rust backend.  Skipping the wrapper there drops ChatML markers from the
prompt prefix and silently corrupts downstream text slicing (evaluators slice
``text[len(prompt):]``).  ``_rust_decode_verified`` therefore probes every
*special* added token (plus a sample of regular ones) once per tokenizer and
enables the Rust path only when both decoders agree.  ``skip_special_tokens``
defaults to ``False``, matching the no-kwarg call sites (and transformers v5's
default), so ChatML markers in the prompt prefix are preserved and the prefix
round-trips exactly.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence

__all__ = [
    "fast_batch_decode",
    "fast_decode",
]

# tokenizer -> bool: whether its raw Rust backend decodes added tokens
# identically to the HuggingFace wrapper.
_RUST_DECODE_VERIFIED: weakref.WeakKeyDictionary[object, bool] = weakref.WeakKeyDictionary()


def _rust_backend(tokenizer):
    """Return the native ``tokenizers.Tokenizer`` if available, else None."""
    return getattr(tokenizer, "_tokenizer", None)


def _rust_decode_verified(tokenizer, backend) -> bool:
    """True iff the raw Rust backend decodes added tokens like the HF wrapper.

    Probed once per tokenizer instance (cached weakly).  Every *special*
    added token is compared (these are exactly the ids whose handling can
    diverge), plus a small sample of regular added tokens, under both
    ``skip_special_tokens`` values.
    """
    try:
        cached = _RUST_DECODE_VERIFIED.get(tokenizer)
    except TypeError:
        cached = None  # non-weakrefable tokenizer: do not cache, probe per call
    if cached is not None:
        return cached

    ok = False
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if decoder:
        special_ids: list[int] = []
        regular_ids: list[int] = []
        for token_id, added_token in decoder.items():
            if getattr(added_token, "special", False):
                special_ids.append(token_id)
            else:
                regular_ids.append(token_id)
        probe_ids = special_ids + regular_ids[:32]
        ok = True
        for token_id in probe_ids:
            ids = [token_id]
            if backend.decode(ids) != tokenizer.decode(ids, skip_special_tokens=False):
                ok = False
                break
            if backend.decode(ids, skip_special_tokens=True) != tokenizer.decode(
                ids, skip_special_tokens=True
            ):
                ok = False
                break

    try:
        _RUST_DECODE_VERIFIED[tokenizer] = ok
    except TypeError:
        pass
    return ok


def fast_decode(tokenizer, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
    """Decode a single sequence, preferring the native Rust ``Tokenizer.decode``.

    ``skip_special_tokens`` defaults to ``False`` to match the no-kwarg call
    sites (and transformers v5's default).  The Rust path is used only after a
    behavior probe proves it matches the wrapper; otherwise this delegates to
    ``tokenizer.decode``.  The keyword is forwarded only when ``True`` on the
    fallback so minimal tokenizer stubs implementing just ``decode(ids)``
    keep working.
    """
    ids = list(token_ids)
    backend = _rust_backend(tokenizer)
    if (
        backend is not None
        and hasattr(backend, "decode")
        and _rust_decode_verified(tokenizer, backend)
    ):
        if skip_special_tokens:
            return backend.decode(ids, skip_special_tokens=True)
        return backend.decode(ids)
    if skip_special_tokens:
        return tokenizer.decode(ids, skip_special_tokens=True)
    return tokenizer.decode(ids)


def fast_batch_decode(
    tokenizer,
    sequences: Sequence[Sequence[int]],
    skip_special_tokens: bool = False,
) -> list[str]:
    """Decode a batch of sequences, preferring the native Rust ``decode_batch``.

    ``skip_special_tokens`` defaults to ``False`` to match the no-kwarg call
    sites (and transformers v5's default).  The Rust path is used only after a
    behavior probe proves it matches the wrapper; otherwise this falls back to
    ``tokenizer.batch_decode`` and then to a per-sequence ``decode`` loop.
    The keyword is only forwarded when ``True`` to stay compatible with
    minimal tokenizer stubs.
    """
    seqs = [list(seq) for seq in sequences]
    backend = _rust_backend(tokenizer)
    if (
        backend is not None
        and hasattr(backend, "decode_batch")
        and _rust_decode_verified(tokenizer, backend)
    ):
        if skip_special_tokens:
            return backend.decode_batch(seqs, skip_special_tokens=True)
        return backend.decode_batch(seqs)
    if hasattr(tokenizer, "batch_decode"):
        if skip_special_tokens:
            return tokenizer.batch_decode(seqs, skip_special_tokens=True)
        return tokenizer.batch_decode(seqs)
    if skip_special_tokens:
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]
    return [tokenizer.decode(seq) for seq in seqs]
