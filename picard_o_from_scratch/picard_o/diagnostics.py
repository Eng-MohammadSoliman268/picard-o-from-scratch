"""Diagnostics and validation helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .backend import scalar_to_bool, scalar_to_float, to_numpy


def internal_validation(W, Y, Z, xp):
    N = W.shape[0]
    I = xp.eye(N, dtype=xp.float64)
    return {
        "orthogonality_error": scalar_to_float(xp.max(xp.abs(W @ W.T - I))),
        "Y_equals_WZ_error": scalar_to_float(xp.max(xp.abs(Y - W @ Z))),
        "W_finite": scalar_to_bool(xp.all(xp.isfinite(W))),
        "Y_finite": scalar_to_bool(xp.all(xp.isfinite(Y))),
    }


def source_correlation_matrix(Y_est, S_ref):
    Y = np.asarray(to_numpy(Y_est), dtype=np.float64)
    S = np.asarray(to_numpy(S_ref), dtype=np.float64)
    if Y.shape[1] != S.shape[1]:
        raise ValueError("Y_est and S_ref must have the same number of samples.")

    Y = Y - Y.mean(axis=1, keepdims=True)
    S = S - S.mean(axis=1, keepdims=True)
    Y = Y / Y.std(axis=1, keepdims=True)
    S = S / S.std(axis=1, keepdims=True)
    return (Y @ S.T) / Y.shape[1]


def match_components(Y_a, Y_b):
    C = source_correlation_matrix(Y_a, Y_b)
    rows, cols = linear_sum_assignment(-np.abs(C))
    matched = C[rows, cols]
    return {
        "correlation_matrix": C,
        "rows": rows,
        "cols": cols,
        "matched_correlation": matched,
        "mean_abs_correlation": float(np.mean(np.abs(matched))),
        "median_abs_correlation": float(np.median(np.abs(matched))),
        "min_abs_correlation": float(np.min(np.abs(matched))),
    }
