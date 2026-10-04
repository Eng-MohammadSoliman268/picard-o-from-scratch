"""
Orthogonal Picard ICA — standalone validated implementation
===========================================================

CPU/GPU backends:
    backend="numpy" -> NumPy / CPU
    backend="cupy"  -> CuPy / CUDA GPU
    backend="auto"  -> GPU when available, otherwise CPU

Input convention:
    X.shape == (n_channels, n_samples)

High-level use:
    from picard_o_full import fit_picard_o
    result = fit_picard_o(X, backend="auto")
    Y = result.Y
    B = result.B_total

This implementation was developed from the Orthogonal Picard mathematics and
validated against python-picard with identical initialization on synthetic data
and 120-dimensional HBN EEG.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm as scipy_expm

try:
    import cupy as cp  # type: ignore
    CUPY_IMPORTED = True
    try:
        GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        GPU_AVAILABLE = False
except Exception:
    cp = None  # type: ignore
    CUPY_IMPORTED = False
    GPU_AVAILABLE = False

try:
    if CUPY_IMPORTED:
        from cupyx.scipy.linalg import expm as cupy_expm  # type: ignore
        CUPY_EXPM_AVAILABLE = True
    else:
        cupy_expm = None
        CUPY_EXPM_AVAILABLE = False
except Exception:
    cupy_expm = None
    CUPY_EXPM_AVAILABLE = False


# -----------------------------------------------------------------------------
# Backend helpers
# -----------------------------------------------------------------------------

def get_backend(backend: str = "auto"):
    backend = backend.lower()
    if backend == "auto":
        if GPU_AVAILABLE:
            return cp, "cupy"
        return np, "numpy"
    if backend == "numpy":
        return np, "numpy"
    if backend == "cupy":
        if not CUPY_IMPORTED:
            raise RuntimeError("CuPy is not installed.")
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy is installed but no CUDA GPU is available.")
        return cp, "cupy"
    raise ValueError("backend must be 'auto', 'numpy', or 'cupy'.")


def to_backend(x, xp):
    if xp is np:
        return np.asarray(x, dtype=np.float64)
    return cp.asarray(x, dtype=cp.float64)


def to_numpy(x):
    if CUPY_IMPORTED and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def scalar_to_float(x) -> float:
    if CUPY_IMPORTED and isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)
    return float(np.asarray(x))


def scalar_to_bool(x) -> bool:
    if CUPY_IMPORTED and isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)
    return bool(np.asarray(x))


def sync_backend(xp) -> None:
    if xp is not np:
        cp.cuda.Stream.null.synchronize()


# -----------------------------------------------------------------------------
# Configuration / result
# -----------------------------------------------------------------------------

@dataclass
class PicardOConfig:
    backend: str = "auto"
    dtype: str = "float64"
    tol: float = 1e-7
    max_iter: int = 500
    lbfgs_memory: int = 7
    lambda_min: float = 1e-2
    ls_tries: int = 10
    rank_tol: float | None = None
    alpha0: float = 1.0


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


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

def center_data(X, xp):
    X = xp.asarray(X, dtype=xp.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_channels, n_samples).")
    if not scalar_to_bool(xp.all(xp.isfinite(X))):
        raise ValueError("X contains NaN or Inf.")
    mean = xp.mean(X, axis=1, keepdims=True)
    return X - mean, mean


def covariance_eigendecomposition(X_centered, xp, rank_tol=None):
    _, n_samples = X_centered.shape
    covariance = (X_centered @ X_centered.T) / n_samples
    eigvals, eigvecs = xp.linalg.eigh(covariance)
    order = xp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    largest_eig = scalar_to_float(eigvals[0])
    if rank_tol is None:
        rank_tol = largest_eig * max(covariance.shape) * np.finfo(np.float64).eps

    rank = int(scalar_to_float(xp.sum(eigvals > rank_tol)))
    return {
        "covariance": covariance,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "rank": rank,
        "rank_tol": float(rank_tol),
    }


def whiten_data(X, xp, rank=None, rank_tol=None):
    Xc, mean = center_data(X, xp=xp)
    eig = covariance_eigendecomposition(Xc, xp=xp, rank_tol=rank_tol)
    estimated_rank = eig["rank"]
    if rank is None:
        rank = estimated_rank
    if rank < 1:
        raise ValueError("Rank must be >= 1.")
    if rank > estimated_rank:
        raise ValueError(
            f"Requested rank={rank}, but estimated numerical rank={estimated_rank}."
        )

    eigvals_r = eig["eigvals"][:rank]
    eigvecs_r = eig["eigvecs"][:, :rank]
    whitening = xp.diag(1.0 / xp.sqrt(eigvals_r)) @ eigvecs_r.T
    dewhitening = eigvecs_r @ xp.diag(xp.sqrt(eigvals_r))
    Z = whitening @ Xc

    return {
        "Z": Z,
        "mean": mean,
        "whitening": whitening,
        "dewhitening": dewhitening,
        "covariance": eig["covariance"],
        "eigvals": eig["eigvals"],
        "eigvecs": eig["eigvecs"],
        "estimated_rank": estimated_rank,
        "used_rank": rank,
        "rank_tol": eig["rank_tol"],
    }


def validate_whitening(Z, xp):
    n_components, n_samples = Z.shape
    C_Z = (Z @ Z.T) / n_samples
    I = xp.eye(n_components, dtype=xp.float64)
    relative_error = xp.linalg.norm(C_Z - I, ord="fro") / xp.linalg.norm(I, ord="fro")
    max_abs_error = xp.max(xp.abs(C_Z - I))
    return {
        "relative_error": scalar_to_float(relative_error),
        "max_abs_error": scalar_to_float(max_abs_error),
    }


# -----------------------------------------------------------------------------
# Tanh density
# -----------------------------------------------------------------------------

class TanhDensity:
    def __init__(self, xp):
        self.xp = xp

    def loss_elementwise(self, Y):
        xp = self.xp
        Y = xp.asarray(Y, dtype=xp.float64)
        abs_Y = xp.abs(Y)
        return (
            abs_Y
            + xp.log1p(xp.exp(-2.0 * abs_Y))
            - xp.log(xp.asarray(2.0, dtype=xp.float64))
        )

    def score_and_der(self, Y):
        xp = self.xp
        Y = xp.asarray(Y, dtype=xp.float64)
        psi = xp.tanh(Y)
        return psi, 1.0 - psi**2

    def loss(self, Y) -> float:
        xp = self.xp
        return scalar_to_float(xp.sum(xp.mean(self.loss_elementwise(Y), axis=1)))


# -----------------------------------------------------------------------------
# Gradient / Hessian approximation
# -----------------------------------------------------------------------------

def compute_relative_gradient(Y, density, xp):
    Y = xp.asarray(Y, dtype=xp.float64)
    _, T = Y.shape
    psiY, psidY = density.score_and_der(Y)
    G_raw = (psiY @ Y.T) / T
    G_ortho = (G_raw - G_raw.T) / 2.0
    return {
        "psiY": psiY,
        "psidY": psidY,
        "G_raw": G_raw,
        "G_ortho": G_ortho,
        "gradient_norm": scalar_to_float(xp.max(xp.abs(G_ortho))),
    }


def compute_hessian_approximation(G_raw, psidY, xp, lambda_min=1e-2):
    psi_der_mean = xp.mean(psidY, axis=1)
    G_diag = xp.diag(G_raw).copy()
    h_unregularized = 0.5 * (
        psi_der_mean[:, None]
        + psi_der_mean[None, :]
        - G_diag[:, None]
        - G_diag[None, :]
    )
    h = xp.maximum(h_unregularized, xp.asarray(lambda_min, dtype=xp.float64))
    return {
        "h_unregularized": h_unregularized,
        "h": h,
        "regularized_mask": h_unregularized < lambda_min,
    }


# -----------------------------------------------------------------------------
# L-BFGS
# -----------------------------------------------------------------------------

class LBFGSMemory:
    def __init__(self, max_size=7):
        self.max_size = int(max_size)
        self.s_list = []
        self.dg_list = []
        self.rho_list = []

    def __len__(self):
        return len(self.s_list)

    def clear(self):
        self.s_list.clear()
        self.dg_list.clear()
        self.rho_list.clear()

    def push(self, step, G_new, G_old, xp):
        delta_G = G_new - G_old
        curvature = scalar_to_float(xp.sum(step * delta_G))
        if not np.isfinite(curvature) or abs(curvature) < 1e-15:
            return False
        self.s_list.append(step.copy())
        self.dg_list.append(delta_G.copy())
        self.rho_list.append(1.0 / curvature)
        if len(self.s_list) > self.max_size:
            self.s_list.pop(0)
            self.dg_list.pop(0)
            self.rho_list.pop(0)
        return True


def compute_lbfgs_direction(G, h, memory, xp):
    q = G.copy()
    alpha_list = []

    for step, delta_G, rho in zip(
        reversed(memory.s_list), reversed(memory.dg_list), reversed(memory.rho_list)
    ):
        alpha = scalar_to_float(rho * xp.sum(step * q))
        alpha_list.append(alpha)
        q = q - alpha * delta_G

    z = q / h
    z = (z - z.T) / 2.0

    for step, delta_G, rho, alpha in zip(
        memory.s_list, memory.dg_list, memory.rho_list, reversed(alpha_list)
    ):
        beta = scalar_to_float(rho * xp.sum(delta_G * z))
        z = z + (alpha - beta) * step

    return -z


# -----------------------------------------------------------------------------
# Matrix exponential / line search
# -----------------------------------------------------------------------------

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
    current_loss,
    ls_tries=10,
    alpha0=1.0,
):
    alpha = float(alpha0)
    history = []

    for attempt in range(ls_tries):
        R = matrix_exponential(alpha * direction, xp=xp)
        Y_new = R @ Y
        W_new = R @ W
        new_loss = density.loss(Y_new)
        improved = np.isfinite(new_loss) and new_loss < current_loss
        history.append(
            {"attempt": attempt, "alpha": alpha, "loss": new_loss, "improved": bool(improved)}
        )
        if improved:
            return {
                "converged": True,
                "alpha": alpha,
                "step": alpha * direction,
                "Y_new": Y_new,
                "W_new": W_new,
                "new_loss": new_loss,
                "attempts": attempt + 1,
                "history": history,
            }
        alpha *= 0.5

    return {"converged": False, "history": history}


# -----------------------------------------------------------------------------
# Validated Picard-O core
# -----------------------------------------------------------------------------

def picard_o_core(Z, xp, config, density=None, verbose=False, w_init=None):
    Z = xp.asarray(Z, dtype=xp.float64)
    N, _ = Z.shape
    density = density or TanhDensity(xp)

    if w_init is None:
        W = xp.eye(N, dtype=xp.float64)
    else:
        W = xp.asarray(w_init, dtype=xp.float64).copy()
        if W.shape != (N, N):
            raise ValueError(f"w_init must have shape {(N, N)}, got {W.shape}.")

    Y = W @ Z
    memory = LBFGSMemory(config.lbfgs_memory)
    current_loss = density.loss(Y)
    initial_loss = current_loss
    history = []
    fallback_count = 0
    accepted_updates = 0
    converged = False

    for iteration in range(config.max_iter):
        grad = compute_relative_gradient(Y, density, xp)
        G = grad["G_ortho"]
        gradient_norm = grad["gradient_norm"]

        if gradient_norm < config.tol:
            converged = True
            history.append(
                {
                    "iteration": iteration,
                    "loss": current_loss,
                    "gradient_norm": gradient_norm,
                    "alpha": 0.0,
                    "memory_size": len(memory),
                    "fallback": False,
                    "accepted": False,
                }
            )
            if verbose:
                print(
                    f"[{iteration:4d}] CONVERGED | loss={current_loss:.12e} | "
                    f"grad={gradient_norm:.3e}"
                )
            break

        h = compute_hessian_approximation(
            grad["G_raw"], grad["psidY"], xp, config.lambda_min
        )["h"]
        direction = compute_lbfgs_direction(G, h, memory, xp)
        used_fallback = False

        line = orthogonal_line_search(
            Y, W, direction, density, xp,
            current_loss=current_loss,
            ls_tries=config.ls_tries,
            alpha0=config.alpha0,
        )

        if not line["converged"]:
            used_fallback = True
            fallback_count += 1
            memory.clear()
            direction = -G
            direction = (direction - direction.T) / 2.0
            line = orthogonal_line_search(
                Y, W, direction, density, xp,
                current_loss=current_loss,
                ls_tries=config.ls_tries,
                alpha0=config.alpha0,
            )

        if not line["converged"]:
            if verbose:
                print(f"[{iteration:4d}] Line search failed after fallback.")
            break

        step = line["step"]
        Y_new = line["Y_new"]
        W_new = line["W_new"]
        new_loss = line["new_loss"]
        grad_new = compute_relative_gradient(Y_new, density, xp)
        memory.push(step, grad_new["G_ortho"], G, xp)

        accepted_updates += 1
        history.append(
            {
                "iteration": iteration,
                "loss": current_loss,
                "new_loss": new_loss,
                "gradient_norm": gradient_norm,
                "next_gradient_norm": grad_new["gradient_norm"],
                "alpha": line["alpha"],
                "memory_size": len(memory),
                "fallback": used_fallback,
                "line_search_attempts": line["attempts"],
                "accepted": True,
            }
        )

        Y, W, current_loss = Y_new, W_new, new_loss

        if verbose and (iteration < 10 or iteration % 10 == 0):
            print(
                f"[{iteration:4d}] loss={current_loss:.12e} | grad={gradient_norm:.3e} | "
                f"alpha={line['alpha']:.3e} | mem={len(memory)} | fallback={used_fallback}"
            )

    final_gradient_norm = compute_relative_gradient(Y, density, xp)["gradient_norm"]
    if final_gradient_norm < config.tol:
        converged = True

    return {
        "Y": Y,
        "W": W,
        "converged": converged,
        "accepted_updates": accepted_updates,
        "history_length": len(history),
        "initial_loss": initial_loss,
        "final_loss": current_loss,
        "final_gradient_norm": final_gradient_norm,
        "fallback_count": fallback_count,
        "history": history,
    }


# -----------------------------------------------------------------------------
# High-level system API
# -----------------------------------------------------------------------------

def fit_picard_o(
    X,
    *,
    backend: str = "auto",
    rank: int | None = None,
    config: PicardOConfig | None = None,
    w_init=None,
    verbose: bool = False,
):
    if config is None:
        config = PicardOConfig(backend=backend)
    elif backend == "auto":
        backend = config.backend

    xp, backend_name = get_backend(backend)
    Xb = to_backend(X, xp)

    sync_backend(xp)
    start = time.perf_counter()

    prep = whiten_data(Xb, xp, rank=rank, rank_tol=config.rank_tol)
    Z = prep["Z"]
    white = validate_whitening(Z, xp)

    core = picard_o_core(
        Z,
        xp,
        config,
        density=TanhDensity(xp),
        verbose=verbose,
        w_init=w_init,
    )

    W = core["W"]
    Y = core["Y"]
    B_total = W @ prep["whitening"]

    I = xp.eye(W.shape[0], dtype=xp.float64)
    orthogonality_error = scalar_to_float(xp.max(xp.abs(W @ W.T - I)))
    Y_equals_WZ_error = scalar_to_float(xp.max(xp.abs(Y - W @ Z)))

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
        whitening_relative_error=white["relative_error"],
        whitening_max_error=white["max_abs_error"],
        orthogonality_error=orthogonality_error,
        Y_equals_WZ_error=Y_equals_WZ_error,
    )


if __name__ == "__main__":
    # Small smoke test
    rng = np.random.default_rng(20261004)
    n, T = 5, 20000
    S = rng.laplace(size=(n, T))
    S -= S.mean(axis=1, keepdims=True)
    S /= S.std(axis=1, keepdims=True)

    while True:
        A = rng.normal(size=(n, n))
        if np.isfinite(np.linalg.cond(A)) and np.linalg.cond(A) < 20:
            break

    X = A @ S
    result = fit_picard_o(X, backend="numpy", verbose=False)
    print("converged:", result.converged)
    print("updates:", result.accepted_updates)
    print("final gradient:", result.final_gradient_norm)
    print("orthogonality error:", result.orthogonality_error)
