"""Limited-memory BFGS with the Picard Hessian preconditioner."""

from __future__ import annotations

import numpy as np

from .backend import scalar_to_float


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
        step = xp.asarray(step, dtype=xp.float64)
        G_new = xp.asarray(G_new, dtype=xp.float64)
        G_old = xp.asarray(G_old, dtype=xp.float64)

        delta_G = G_new - G_old
        curvature = xp.sum(step * delta_G)
        curvature_float = scalar_to_float(curvature)

        if not np.isfinite(curvature_float) or abs(curvature_float) < 1e-15:
            return False

        rho = 1.0 / curvature_float
        self.s_list.append(step.copy())
        self.dg_list.append(delta_G.copy())
        self.rho_list.append(rho)

        if len(self.s_list) > self.max_size:
            self.s_list.pop(0)
            self.dg_list.pop(0)
            self.rho_list.pop(0)
        return True


def compute_lbfgs_direction(G, h, memory: LBFGSMemory, xp):
    G = xp.asarray(G, dtype=xp.float64)
    h = xp.asarray(h, dtype=xp.float64)

    q = G.copy()
    alpha_list = []

    for step, delta_G, rho in zip(
        reversed(memory.s_list),
        reversed(memory.dg_list),
        reversed(memory.rho_list),
    ):
        alpha = rho * xp.sum(step * q)
        alpha_float = scalar_to_float(alpha)
        alpha_list.append(alpha_float)
        q = q - alpha_float * delta_G

    z = q / h
    z = (z - z.T) / 2.0

    for step, delta_G, rho, alpha in zip(
        memory.s_list,
        memory.dg_list,
        memory.rho_list,
        reversed(alpha_list),
    ):
        beta = rho * xp.sum(delta_G * z)
        beta_float = scalar_to_float(beta)
        z = z + (alpha - beta_float) * step

    return {
        "direction": -z,
        "q": q,
        "preconditioned_q": z,
        "memory_size": len(memory),
    }
