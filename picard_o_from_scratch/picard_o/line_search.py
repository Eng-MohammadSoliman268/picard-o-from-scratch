"""Orthogonal matrix exponential update and backtracking line search."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm as scipy_expm

from .backend import CUPY_IMPORTED, cp

try:  # pragma: no cover - environment dependent
    if CUPY_IMPORTED:
        from cupyx.scipy.linalg import expm as cupy_expm  # type: ignore
        CUPY_EXPM_AVAILABLE = True
    else:
        cupy_expm = None
        CUPY_EXPM_AVAILABLE = False
except Exception:  # pragma: no cover
    cupy_expm = None
    CUPY_EXPM_AVAILABLE = False


def matrix_exponential(A, xp):
    if xp is np:
        return scipy_expm(np.asarray(A, dtype=np.float64))
    if not CUPY_EXPM_AVAILABLE:
        raise RuntimeError(
            "cupyx.scipy.linalg.expm is unavailable. Install a compatible CuPy build."
        )
    return cupy_expm(cp.asarray(A, dtype=cp.float64))


def orthogonal_line_search(
    Y,
    W,
    direction,
    density,
    xp,
    current_loss=None,
    ls_tries=10,
    alpha0=1.0,
):
    Y = xp.asarray(Y, dtype=xp.float64)
    W = xp.asarray(W, dtype=xp.float64)
    D = xp.asarray(direction, dtype=xp.float64)

    if current_loss is None:
        current_loss = density.loss(Y)

    alpha = float(alpha0)
    history = []

    for attempt in range(ls_tries):
        R = matrix_exponential(alpha * D, xp=xp)
        Y_new = R @ Y
        W_new = R @ W
        new_loss = density.loss(Y_new)
        finite = np.isfinite(new_loss)
        improved = finite and new_loss < current_loss

        history.append(
            {
                "attempt": attempt,
                "alpha": alpha,
                "loss": new_loss,
                "improved": bool(improved),
            }
        )

        if improved:
            return {
                "converged": True,
                "alpha": alpha,
                "step": alpha * D,
                "transform": R,
                "Y_new": Y_new,
                "W_new": W_new,
                "new_loss": new_loss,
                "old_loss": current_loss,
                "attempts": attempt + 1,
                "history": history,
            }
        alpha *= 0.5

    return {
        "converged": False,
        "alpha": alpha,
        "step": None,
        "transform": None,
        "Y_new": None,
        "W_new": None,
        "new_loss": None,
        "old_loss": current_loss,
        "attempts": ls_tries,
        "history": history,
    }
