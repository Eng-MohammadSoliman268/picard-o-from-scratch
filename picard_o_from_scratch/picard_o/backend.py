"""Backend helpers for NumPy / CuPy execution."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # type: ignore
    CUPY_IMPORTED = True
    try:
        GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        GPU_AVAILABLE = False
except Exception:  # pragma: no cover - depends on environment
    cp = None  # type: ignore
    CUPY_IMPORTED = False
    GPU_AVAILABLE = False


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
