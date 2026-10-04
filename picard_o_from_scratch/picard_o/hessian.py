"""Picard-O Hessian approximation / preconditioner."""

from __future__ import annotations


def compute_hessian_approximation(G_raw, psidY, xp, lambda_min=1e-2):
    G_raw = xp.asarray(G_raw, dtype=xp.float64)
    psidY = xp.asarray(psidY, dtype=xp.float64)

    psi_der_mean = xp.mean(psidY, axis=1)
    G_diag = xp.diag(G_raw).copy()

    h_unregularized = 0.5 * (
        psi_der_mean[:, None]
        + psi_der_mean[None, :]
        - G_diag[:, None]
        - G_diag[None, :]
    )

    h = xp.maximum(
        h_unregularized,
        xp.asarray(lambda_min, dtype=xp.float64),
    )

    return {
        "psi_der_mean": psi_der_mean,
        "G_diag": G_diag,
        "h_unregularized": h_unregularized,
        "h": h,
        "regularized_mask": h_unregularized < lambda_min,
    }
