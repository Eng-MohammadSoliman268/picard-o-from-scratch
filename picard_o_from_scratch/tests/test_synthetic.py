import numpy as np

from picard_o import PicardOConfig, fit_picard_o, to_numpy
from picard_o.diagnostics import match_components


def build_problem(seed=20261004, n=5, T=20000):
    rng = np.random.default_rng(seed)
    S = rng.laplace(size=(n, T))
    S -= S.mean(axis=1, keepdims=True)
    S /= S.std(axis=1, keepdims=True)

    while True:
        A = rng.normal(size=(n, n))
        cond = np.linalg.cond(A)
        if np.isfinite(cond) and cond < 20:
            break
    return A @ S, S


def test_synthetic_recovery_numpy():
    X, S = build_problem()
    config = PicardOConfig(backend="numpy", tol=1e-7, max_iter=500)
    result = fit_picard_o(X, config=config)

    assert result.converged
    assert result.final_gradient_norm < config.tol
    assert result.orthogonality_error < 1e-10
    assert result.Y_equals_WZ_error < 1e-10

    matched = match_components(to_numpy(result.Y), S)
    assert matched["mean_abs_correlation"] > 0.999
    assert matched["min_abs_correlation"] > 0.998
