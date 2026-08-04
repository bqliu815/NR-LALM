# Frozen protocol

- Problem dimension: `n = 100`; equality constraints: `m = 20`.
- Problem seed: 6101; direction seed: 6201; directions: 32.
- Penalties: `beta = 3e6`, `rho = 1.92e8`.
- Initial multiplier norm: 1049.
- Grid and local step ranges: exactly those in `configs/paper_v1.toml`.
- Plot names: NR-LALM and NR-LALM+SOC.
- Output: white-background vector PDF and 300-dpi PNG, with the same physical
  size and typography used by the paper's Section 5.3 figure.

The run passes only if the base certified region is nonempty and contained in
the corrected region, the fixed strict point belongs only to the corrected
region, the fitted orders lie in `[1.9, 2.1]` and `[3.8, 4.2]`, all fits have
`R^2 >= 0.999`, and the correction solves meet the stated residual tolerance.
