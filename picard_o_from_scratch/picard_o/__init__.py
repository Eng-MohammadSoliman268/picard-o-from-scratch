"""Orthogonal Picard ICA from scratch with NumPy/CuPy backends."""

from .api import PicardOResult, fit_picard_o
from .config import PicardOConfig
from .core import picard_o_core
from .backend import get_backend, to_numpy

__all__ = [
    "PicardOConfig",
    "PicardOResult",
    "fit_picard_o",
    "picard_o_core",
    "get_backend",
    "to_numpy",
]

__version__ = "1.0.0"
