import numpy as np
import pytest

from picard_o.backend import CUPY_IMPORTED, GPU_AVAILABLE, cp
from picard_o.preprocessing import whiten_data, validate_whitening


def test_numpy_whitening():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 10000))
    prep = whiten_data(X, xp=np)
    check = validate_whitening(prep["Z"], xp=np)
    assert check["relative_error"] < 1e-10


@pytest.mark.skipif(not (CUPY_IMPORTED and GPU_AVAILABLE), reason="CUDA GPU unavailable")
def test_numpy_cupy_whitening_consistency():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 5000))
    prep_np = whiten_data(X, xp=np)
    prep_cp = whiten_data(cp.asarray(X), xp=cp)
    eig_np = np.asarray(prep_np["eigvals"])
    eig_cp = cp.asnumpy(prep_cp["eigvals"])
    assert np.max(np.abs(eig_np - eig_cp)) < 1e-10
