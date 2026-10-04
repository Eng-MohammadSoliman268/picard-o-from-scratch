# Validation Record

## Synthetic validation

Validated development run:

- 5 independent non-Gaussian sources.
- Mean matched source correlation: `0.9998366110984312`.
- Minimum matched source correlation: `0.9995768108762855`.
- Our implementation vs official Picard-O after sign/permutation alignment: mean `|r| = 0.999999999999529`.

## HBN EEG validation

Dataset used during development:

- Raw shape after selected bad-channel removal: `121 x 201737`.
- Estimated numerical rank: `120`.
- Whitened shape: `120 x 201737`.
- Backend: CuPy.
- Converged: yes.
- Accepted updates: `292`.
- Final gradient norm: `9.663381700386608e-08`.
- Initial loss: `42.58421372111494`.
- Final loss: `27.81144126031796`.
- Fallback count: `0`.
- W orthogonality max error: `8.79296635503124e-14`.
- `Y = WZ` max error: `6.892264536872972e-13`.
- Development runtime: `52.02329607700085 s`.

## Official Picard-O equivalence on HBN

The comparison was made with the same whitened input and identical initialization `W0 = I`.

- Official iterations: `292`.
- Our accepted updates: `292`.
- Our final gradient: `9.663381682345484e-08`.
- Official final gradient: `9.685612465532012e-08`.
- Our final loss: `27.811441260317956`.
- Official final loss: `27.811441260315352`.
- Loss difference: `2.604139126560767e-12`.
- Mean matched component `|r|`: `0.9999999999997569`.
- Median matched component `|r|`: `0.9999999999999293`.
- Minimum matched component `|r|`: `0.9999999999965969`.
- Components with `|r| >= 0.999`: `120 / 120`.
- Aligned W relative error: `6.974006686438216e-07`.
- Aligned Y relative error: `6.974006687118632e-07`.

## Interpretation

The core is frozen as the validated v1.0.0 mathematical implementation. Changes to optimization equations, gradient/Hessian definitions, L-BFGS updates, line search, or orthogonal exponential updates should be revalidated against these checks.
