"""High-level API for system integration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .backend import get_backend, sync_backend, to_backend
from .config import PicardOConfig
from .core import picard_o_core
from .densities import TanhDensity
from .diagnostics import internal_validation
from .preprocessing import validate_whitening, whiten_data


@dataclass
class PicardOResult:
    Y: Any
    W: Any
    B_total: Any
    mean: Any
    whitening: Any
    dewhitening: Any
    rank: int
    estimated_rank: int
    converged: bool
    accepted_updates: int
    history_length: int
    initial_loss: float
    final_loss: float
    final_gradient_norm: float
    fallback_count: int
    runtime_seconds: float
    backend: str
    history: list
    whitening_relative_error: float
    whitening_max_error: float
    orthogonality_error: float
    Y_equals_WZ_error: float


def fit_picard_o(
    X,
    *,
    backend: str = "auto",
    rank: int | None = None,
    config: PicardOConfig | None = None,
    w_init=None,
    verbose: bool = False,
):
    """Fit Orthogonal Picard ICA to raw channels x samples data.

    The function centers, rank-reduces, whitens, runs the validated
    Orthogonal Picard core, and returns the total unmixing matrix.
    """

    if config is None:
        config = PicardOConfig(backend=backend)
    else:
        backend = config.backend if backend == "auto" else backend

    xp, backend_name = get_backend(backend)
    X_backend = to_backend(X, xp)

    sync_backend(xp)
    start = time.perf_counter()

    prep = whiten_data(
        X_backend,
        xp=xp,
        rank=rank,
        rank_tol=config.rank_tol,
    )
    Z = prep["Z"]
    white_check = validate_whitening(Z, xp=xp)

    core = picard_o_core(
        Z,
        xp=xp,
        config=config,
        density=TanhDensity(xp),
        verbose=verbose,
        w_init=w_init,
    )

    W = core["W"]
    Y = core["Y"]
    B_total = W @ prep["whitening"]
    diag = internal_validation(W, Y, Z, xp=xp)

    sync_backend(xp)
    runtime = time.perf_counter() - start

    return PicardOResult(
        Y=Y,
        W=W,
        B_total=B_total,
        mean=prep["mean"],
        whitening=prep["whitening"],
        dewhitening=prep["dewhitening"],
        rank=prep["used_rank"],
        estimated_rank=prep["estimated_rank"],
        converged=core["converged"],
        accepted_updates=core["accepted_updates"],
        history_length=core["history_length"],
        initial_loss=core["initial_loss"],
        final_loss=core["final_loss"],
        final_gradient_norm=core["final_gradient_norm"],
        fallback_count=core["fallback_count"],
        runtime_seconds=runtime,
        backend=backend_name,
        history=core["history"],
        whitening_relative_error=white_check["relative_error"],
        whitening_max_error=white_check["max_abs_error"],
        orthogonality_error=diag["orthogonality_error"],
        Y_equals_WZ_error=diag["Y_equals_WZ_error"],
    )
