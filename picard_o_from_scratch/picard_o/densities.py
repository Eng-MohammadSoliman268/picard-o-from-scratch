"""Density / contrast functions used by the validated Picard-O core."""

from __future__ import annotations

from .backend import scalar_to_float


class TanhDensity:
    """Validated tanh contrast.

    loss element: log(cosh(y))
    psi(y): tanh(y)
    psi'(y): 1 - tanh(y)^2
    """

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
        psi_der = 1.0 - psi**2
        return psi, psi_der

    def loss(self, Y) -> float:
        xp = self.xp
        L = self.loss_elementwise(Y)
        return scalar_to_float(xp.sum(xp.mean(L, axis=1)))
