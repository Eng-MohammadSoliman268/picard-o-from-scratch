# Orthogonal Picard ICA — NumPy + CuPy

A from-scratch implementation of **Orthogonal Picard ICA (Picard-O)** for batch ICA, designed for scientific signal-processing systems and validated on both synthetic mixtures and real HBN EEG.

## Status

**Validated core — v1.0.0**

The implementation was validated in the development workflow against the official `python-picard` implementation using identical initialization.

Validation highlights:

- Synthetic 5-source problem: mean matched source correlation ≈ **0.99984**.
- Synthetic output vs official Picard-O: mean matched component correlation ≈ **1.0**.
- HBN EEG input: **121 channels × 201,737 samples**.
- HBN numerical rank after preprocessing: **120**.
- HBN convergence: final max orthogonal gradient ≈ **9.66e-8** with tolerance **1e-7**.
- HBN same-initialization comparison vs official Picard-O:
  - 120 / 120 components with `|r| >= 0.999`.
  - mean `|r| ≈ 0.9999999999997569`.
  - minimum `|r| ≈ 0.9999999999965969`.
  - final loss difference ≈ `2.60e-12`.
  - accepted updates: **292**, matching official iterations **292**.
- Validated CuPy HBN runtime during development: ~**52 s** versus ~**427 s** for the CPU official comparison in that environment. This is a hardware/backend comparison, not an algorithmic benchmark.

## Algorithm

The implementation follows this pipeline:

```text
X
 -> center
 -> covariance eigendecomposition
 -> rank reduction
 -> whitening Z
 -> tanh score psi and psi'
 -> relative orthogonal gradient
 -> Picard Hessian approximation
 -> preconditioned L-BFGS
 -> backtracking line search
 -> exp(alpha D) orthogonal update
 -> repeat until max|G_ortho| < tol
```

The orthogonal update is

```text
W_new = exp(alpha D) @ W
```

with skew-symmetric `D`, so the update stays on the orthogonal manifold.

## Installation

CPU only:

```bash
pip install -e .
```

For development:

```bash
pip install -r requirements-dev.txt
```

GPU support requires a CuPy build compatible with the installed CUDA version. For example, on some CUDA 12 environments:

```bash
pip install cupy-cuda12x
```

Do not install a CuPy CUDA package blindly; match it to the CUDA runtime on the target machine.

## Quick start

```python
import numpy as np
from picard_o import PicardOConfig, fit_picard_o, to_numpy

# X shape: (channels, samples)
X = np.load("data.npy")

config = PicardOConfig(
    backend="auto",      # "numpy", "cupy", or "auto"
    tol=1e-7,
    max_iter=500,
    lbfgs_memory=7,
    lambda_min=1e-2,
    ls_tries=10,
)

result = fit_picard_o(X, config=config)

print(result.converged)
print(result.rank)
print(result.accepted_updates)
print(result.final_gradient_norm)

Y = to_numpy(result.Y)
B_total = to_numpy(result.B_total)
```

## Main outputs

`fit_picard_o(...)` returns a `PicardOResult` with:

- `Y`: estimated independent components.
- `W`: orthogonal unmixing matrix in whitened space.
- `B_total`: total unmixing matrix from centered raw channels to ICs.
- `mean`: channel mean used for centering.
- `whitening`, `dewhitening`: preprocessing transforms.
- `rank`, `estimated_rank`.
- `converged`.
- `accepted_updates`.
- `final_gradient_norm`.
- `initial_loss`, `final_loss`.
- `fallback_count`.
- `runtime_seconds`.
- whitening and orthogonality diagnostics.
- full iteration `history`.

To apply the fitted transform to data using the same channel ordering and preprocessing convention:

```python
X_centered = X - to_numpy(result.mean)
Y = to_numpy(result.B_total) @ X_centered
```

## Backend selection

```python
result = fit_picard_o(X, backend="numpy")
```

or

```python
result = fit_picard_o(X, backend="cupy")
```

or automatic selection:

```python
result = fit_picard_o(X, backend="auto")
```

The same core equations are used for NumPy and CuPy.

## Reproducible initialization

The validated development runs used identity initialization:

```text
W0 = I
```

This is the default here. For controlled experiments, `w_init` can be supplied explicitly.

Initialization matters on high-dimensional real EEG because ICA is a non-convex optimization problem. Comparisons against another implementation should use the same initialization before interpreting component-by-component differences.

## Repository layout

```text
picard_o_from_scratch/
├── picard_o/
│   ├── backend.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── densities.py
│   ├── gradient.py
│   ├── hessian.py
│   ├── lbfgs.py
│   ├── line_search.py
│   ├── core.py
│   ├── api.py
│   └── diagnostics.py
├── examples/
│   ├── system_integration.py
│   └── hbn_example.py
├── tests/
│   ├── test_backend.py
│   ├── test_synthetic.py
│   └── test_official_optional.py
├── picard_o_full.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

## Standalone integration

If the target system should not install the package layout, use:

```text
picard_o_full.py
```

It contains the full validated algorithm in one Python file.

```python
from picard_o_full import fit_picard_o

result = fit_picard_o(X, backend="cupy")
```

## Tests

Run:

```bash
pytest -q
```

The official-equivalence test is optional and runs only if `python-picard` is installed.

## Scope

This repository contains the validated **batch Orthogonal Picard ICA** implementation. It is not an online ICA algorithm. For a real-time/streaming pipeline, use a streaming method such as the separately developed ORICA path rather than treating Picard-O as an online update rule.

## Scientific background

The implementation is based on the Orthogonal Picard / Hessian-preconditioned ICA approach introduced by Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort and collaborators. The official `python-picard` project was used as an independent numerical validation target; this repository is organized as a separate from-scratch system implementation.
