# Section 5.2 deterministic benchmark protocol

The experiment compares NR-LALM, NR-LALM+SOC, strict L-AL, and IPOPT on
the 15 LIBSVM binary-classification data sets listed in
`configs/paper_stage_b_v2.json`.

For every data set, the objective is binary logistic loss subject to ten
seeded affine equalities and one sphere equality. Explicit sparse zeros are
discarded and every nonzero sample row is normalized to unit Euclidean norm.
Files with at most 20,000 observations are used in full; larger files use a
uniform 20,000-row Algorithm-R reservoir sample with seed 20260731. A
stratified 80/20 split with the same seed defines the optimization and
auxiliary test portions.

All four methods start from the same exactly feasible point and use the same
independent squared pair-KKT residual. The primary target is
`R_k^2 <= 1e-8`; `1e-10` and `1e-12` are retained as sensitivity targets.
NR-LALM, NR-LALM+SOC, and L-AL use the common parameters in
`configs/paper_stage_b_v2.json`. IPOPT uses the PardisoMKL sparse direct
solver through an IPOPT build linked to Intel oneAPI Math Kernel Library.

Each data set is run in one exclusive allocation. Every method is executed
in a fresh Python process under eight balanced orders, so each method
occupies every execution position twice. Numerical-library thread counts are
fixed to one. The paper table reports median first-hit wall time over the
eight runs; failures and 1,800-second timeouts remain in the denominator.

The accompanying data manifest contains the download and deterministic
sampling information required for these 15 paper data sets.
