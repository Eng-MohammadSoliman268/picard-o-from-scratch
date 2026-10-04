"""Minimal system-integration example."""

import numpy as np
from picard_o import PicardOConfig, fit_picard_o, to_numpy

# X must be channels x samples.
X = np.load("your_multichannel_data.npy")

config = PicardOConfig(
    backend="auto",
    tol=1e-7,
    max_iter=500,
    lbfgs_memory=7,
    lambda_min=1e-2,
    ls_tries=10,
)

result = fit_picard_o(X, config=config)

print("backend:", result.backend)
print("rank:", result.rank)
print("converged:", result.converged)
print("accepted updates:", result.accepted_updates)
print("final gradient:", result.final_gradient_norm)
print("runtime [s]:", result.runtime_seconds)

Y_cpu = to_numpy(result.Y)
B_total_cpu = to_numpy(result.B_total)

np.save("ica_sources.npy", Y_cpu)
np.save("ica_unmixing.npy", B_total_cpu)
