"""Optional equivalence test when python-picard is installed."""

import importlib.util
import numpy as np
import pytest

from picard_o import PicardOConfig, fit_picard_o, to_numpy
from picard_o.diagnostics import match_components

HAS_PICARD = importlib.util.find_spec("picard") is not None


@pytest.mark.skipif(not HAS_PICARD, reason="python-picard not installed")
def test_against_official_same_initialization():
    from picard import picard as official_picard

    rng = np.random.default_rng(7)
    n, T = 5, 12000
    S = rng.laplace(size=(n, T))
    S -= S.mean(axis=1, keepdims=True)
    A = rng.normal(size=(n, n))
    X = A @ S

    ours = fit_picard_o(X, config=PicardOConfig(backend="numpy", tol=1e-7))
    Z = ours.whitening @ (X - to_numpy(ours.mean))
    W0 = np.eye(n)
    _, W_off, Y_off = official_picard(
        Z,
        fun="tanh",
        ortho=True,
        extended=False,
        whiten=False,
        centering=False,
        max_iter=500,
        tol=1e-7,
        m=7,
        ls_tries=10,
        lambda_min=1e-2,
        w_init=W0,
        verbose=False,
    )

    matched = match_components(to_numpy(ours.Y), Y_off)
    assert matched["mean_abs_correlation"] > 0.999999
