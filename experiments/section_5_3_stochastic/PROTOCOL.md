# Frozen stochastic protocol

For each observation `(a_j, y_j)` and class vector `w_i`, the objective is
the finite-sum softmax cross-entropy loss. The deterministic equalities are

```text
c_i(x) = (||w_i||^2 - 1) / 2 = 0
```

for every class. Only objective components are sampled.

Every method starts from the same data-independent construction: four
sphere-projected gradient steps of length 0.5 on a fixed 4,096-observation
subset, followed by block scaling by 0.95 and a zero multiplier. The 20,480
component gradients used by this common preprocessing are reported separately
and excluded from the horizontal axis.

The common budget is 262,144 objective component-gradient evaluations. The
evaluator records 17 checkpoints. NR-LALM and NR-LALM+SOC use identical
samples, `(rho, beta) = (300, 48)`, and the projected-SPIDER schedule in
`configs/paper_v1.json`. MLALM and S-SQP use the fixed literature-based
parameters in the same file.

For every method and checkpoint, the full-data evaluator computes the
least-squares multiplier and

```text
R_min^2(x) = ||grad f(x) + grad c(x) lambda_hat(x)||^2 + ||c(x)||^2.
```

The figure uses the pointwise arithmetic mean over ten independent streams
per data set. It applies no smoothing, running-best transformation,
uncertainty band, inset, or checkpoint selection.
