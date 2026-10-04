"""Configuration dataclass for Orthogonal Picard ICA."""

from dataclasses import dataclass


@dataclass
class PicardOConfig:
    backend: str = "auto"
    dtype: str = "float64"
    tol: float = 1e-7
    max_iter: int = 500
    lbfgs_memory: int = 7
    lambda_min: float = 1e-2
    ls_tries: int = 10
    rank_tol: float | None = None
    alpha0: float = 1.0
