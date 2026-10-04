"""Validated Orthogonal Picard ICA optimization core."""

from __future__ import annotations

from .backend import scalar_to_float
from .densities import TanhDensity
from .gradient import compute_relative_gradient
from .hessian import compute_hessian_approximation
from .lbfgs import LBFGSMemory, compute_lbfgs_direction
from .line_search import orthogonal_line_search


def picard_o_core(Z, xp, config, density=None, verbose=False, w_init=None):
    Z = xp.asarray(Z, dtype=xp.float64)
    N, _ = Z.shape

    if density is None:
        density = TanhDensity(xp)

    if w_init is None:
        W = xp.eye(N, dtype=xp.float64)
    else:
        W = xp.asarray(w_init, dtype=xp.float64).copy()
        if W.shape != (N, N):
            raise ValueError(f"w_init must have shape {(N, N)}, got {W.shape}.")

    Y = W @ Z
    memory = LBFGSMemory(max_size=config.lbfgs_memory)
    current_loss = density.loss(Y)
    initial_loss = current_loss
    history = []
    converged = False
    fallback_count = 0
    accepted_updates = 0

    for iteration in range(config.max_iter):
        grad = compute_relative_gradient(Y, density=density, xp=xp)
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

        hess = compute_hessian_approximation(
            G_raw=grad["G_raw"],
            psidY=grad["psidY"],
            xp=xp,
            lambda_min=config.lambda_min,
        )
        h = hess["h"]

        lbfgs = compute_lbfgs_direction(G=G, h=h, memory=memory, xp=xp)
        direction = lbfgs["direction"]
        directional_derivative = scalar_to_float(xp.sum(G * direction))
        used_fallback = False

        line = orthogonal_line_search(
            Y=Y,
            W=W,
            direction=direction,
            density=density,
            xp=xp,
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
            directional_derivative = scalar_to_float(xp.sum(G * direction))
            line = orthogonal_line_search(
                Y=Y,
                W=W,
                direction=direction,
                density=density,
                xp=xp,
                current_loss=current_loss,
                ls_tries=config.ls_tries,
                alpha0=config.alpha0,
            )

        if not line["converged"]:
            if verbose:
                print(f"[{iteration:4d}] Line search failed even after fallback.")
            break

        step = line["step"]
        Y_new = line["Y_new"]
        W_new = line["W_new"]
        new_loss = line["new_loss"]
        alpha = line["alpha"]

        grad_new = compute_relative_gradient(Y_new, density=density, xp=xp)
        G_new = grad_new["G_ortho"]
        stored = memory.push(step=step, G_new=G_new, G_old=G, xp=xp)

        accepted_updates += 1
        history.append(
            {
                "iteration": iteration,
                "loss": current_loss,
                "new_loss": new_loss,
                "gradient_norm": gradient_norm,
                "next_gradient_norm": grad_new["gradient_norm"],
                "alpha": alpha,
                "memory_size": len(memory),
                "stored": stored,
                "fallback": used_fallback,
                "directional_derivative": directional_derivative,
                "line_search_attempts": line["attempts"],
                "accepted": True,
            }
        )

        Y = Y_new
        W = W_new
        current_loss = new_loss

        if verbose and (iteration < 10 or iteration % 10 == 0):
            print(
                f"[{iteration:4d}] loss={current_loss:.12e} | "
                f"grad={gradient_norm:.3e} | alpha={alpha:.3e} | "
                f"mem={len(memory)} | fallback={used_fallback}"
            )

    final_grad = compute_relative_gradient(Y, density=density, xp=xp)
    final_gradient_norm = final_grad["gradient_norm"]
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
        "lbfgs_memory": memory,
    }
