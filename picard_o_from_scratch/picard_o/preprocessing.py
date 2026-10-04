"""Centering, rank estimation, whitening and dewhitening."""

from __future__ import annotations

import numpy as np

from .backend import scalar_to_bool, scalar_to_float


def center_data(X, xp):
    X = xp.asarray(X, dtype=xp.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_channels, n_samples).")
    if not scalar_to_bool(xp.all(xp.isfinite(X))):
        raise ValueError("X contains NaN or Inf.")
    mean = xp.mean(X, axis=1, keepdims=True)
    return X - mean, mean


def covariance_eigendecomposition(X_centered, xp, rank_tol=None):
    n_channels, n_samples = X_centered.shape
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
        "covariance": C_Z,
        "relative_error": scalar_to_float(relative_error),
        "max_abs_error": scalar_to_float(max_abs_error),
    }
