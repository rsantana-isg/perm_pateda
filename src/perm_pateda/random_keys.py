"""Utilities for random-key representations of permutations."""

from __future__ import annotations

from typing import Optional

import numpy as np


def random_keys_to_permutation(random_keys: np.ndarray) -> np.ndarray:
    """Convert random keys to a permutation (or population of permutations)."""
    keys = np.asarray(random_keys, dtype=float)
    if keys.ndim == 1:
        return np.argsort(keys, kind="stable").astype(int)
    if keys.ndim == 2:
        return np.argsort(keys, axis=1, kind="stable").astype(int)
    raise ValueError("random_keys must be a 1-D or 2-D array")


def permutation_to_random_keys(
    permutation: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    jitter: float = 1e-9,
) -> np.ndarray:
    """
    Convert permutation(s) to random keys preserving the represented order.

    A tiny jitter can be added to avoid exact duplicates while preserving order.
    """
    perm = np.asarray(permutation, dtype=int)

    def _single_to_keys(one_perm: np.ndarray) -> np.ndarray:
        n = one_perm.size
        if n == 0:
            return np.array([], dtype=float)
        if sorted(one_perm.tolist()) != list(range(n)):
            raise ValueError("Input is not a valid 0-indexed permutation")
        if n > 1:
            base = np.linspace(0.0, 1.0, num=n, dtype=float)
        else:
            base = np.array([0.0], dtype=float)
        out = np.empty(n, dtype=float)
        out[one_perm] = base
        if rng is not None and jitter > 0.0 and n > 1:
            step = 1.0 / float(n - 1)
            eps = min(jitter, step / 4.0)
            out = out + rng.uniform(-eps, eps, size=n)
            out = np.clip(out, 0.0, 1.0)
            # Reassign sorted values to keep exact RK spacing while using jitter only
            # as a tie-breaker when values collide after clipping.
            order = np.argsort(out, kind="stable")
            out[order] = np.linspace(0.0, 1.0, num=n, dtype=float)
        return out

    if perm.ndim == 1:
        return _single_to_keys(perm)
    if perm.ndim == 2:
        return np.vstack([_single_to_keys(p) for p in perm])
    raise ValueError("permutation must be a 1-D or 2-D array")


def random_keys_to_ranks(random_keys: np.ndarray) -> np.ndarray:
    """Convert random keys to 1-based ranks."""
    keys = np.asarray(random_keys, dtype=float)

    def _single_to_ranks(one_keys: np.ndarray) -> np.ndarray:
        n = one_keys.size
        order = np.argsort(one_keys, kind="stable")
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1, dtype=int)
        return ranks

    if keys.ndim == 1:
        return _single_to_ranks(keys)
    if keys.ndim == 2:
        return np.vstack([_single_to_ranks(k) for k in keys])
    raise ValueError("random_keys must be a 1-D or 2-D array")


def _rank_rescale_random_keys(random_keys: np.ndarray) -> np.ndarray:
    """Rescale random keys using the rank-based RK-EDA procedure."""
    keys = np.asarray(random_keys, dtype=float)

    def _single_rescale(one_keys: np.ndarray) -> np.ndarray:
        n = one_keys.size
        if n <= 1:
            return np.zeros_like(one_keys, dtype=float)
        ranks = random_keys_to_ranks(one_keys)
        return (ranks.astype(float) - 1.0) / float(n - 1)

    if keys.ndim == 1:
        return _single_rescale(keys)
    if keys.ndim == 2:
        return np.vstack([_single_rescale(k) for k in keys])
    raise ValueError("random_keys must be a 1-D or 2-D array")


def rescale_random_keys(random_keys: np.ndarray) -> np.ndarray:
    """Public alias for rank-based random-key rescaling."""
    return _rank_rescale_random_keys(random_keys)
