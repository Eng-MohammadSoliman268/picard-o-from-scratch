"""Relative gradient on the orthogonal manifold."""

from __future__ import annotations

from .backend import scalar_to_float


def compute_relative_gradient(Y, density, xp):
    Y = xp.asarray(Y, dtype=xp.float64)
    _, T = Y.shape
    psiY, psidY = density.score_and_der(Y)
    G_raw = (psiY @ Y.T) / T
    G_ortho = (G_raw - G_raw.T) / 2.0
    gradient_norm = xp.max(xp.abs(G_ortho))
    return {
        "psiY": psiY,
        "psidY": psidY,
        "G_raw": G_raw,
        "G_ortho": G_ortho,
        "gradient_norm": scalar_to_float(gradient_norm),
    }
