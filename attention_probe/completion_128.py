"""Isolated EOS continuation cap; the original 48-token decoder is unchanged.

The exact existing decoder code runs with a private copy of its globals whose
single MAX_NEW_TOKENS guard is 128. No module global is changed. Its original
48 default is never used here: the wrapper supplies the total cap explicitly.
"""
from __future__ import annotations

import types

from . import answer_validation as original

MAX_NEW_TOKENS = 128
_NAMESPACE = dict(original.__dict__, MAX_NEW_TOKENS=MAX_NEW_TOKENS)
_DECODER = types.FunctionType(original.run_generation_arm.__code__, _NAMESPACE,
                              original.run_generation_arm.__name__, original.run_generation_arm.__defaults__,
                              original.run_generation_arm.__closure__)
_DECODER.__kwdefaults__ = dict(original.run_generation_arm.__kwdefaults__ or {})


def run_generation_arm(*args, max_new_tokens=MAX_NEW_TOKENS, **kwargs):
    """Fresh prompt/cache, original prefill pulse, at most 128 total selected tokens."""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError('max_new_tokens must be an integer from 1 through 128')
    return _DECODER(*args, max_new_tokens=max_new_tokens, **kwargs)
