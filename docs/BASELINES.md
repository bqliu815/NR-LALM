# Baseline implementations

No third-party source code is bundled. The baseline solvers in this
repository are independent equation-level implementations used by the paper.

- **L-AL** follows L. El Bourkhissi and I. Necoara, *Complexity of a
  Linearized Augmented Lagrangian Method for Nonconvex Minimization with
  Nonlinear Equality Constraints*, arXiv:2301.08345 (2025).
- **IPOPT** is accessed through `cyipopt`; see A. Wächter and L. T. Biegler,
  *On the Implementation of an Interior-Point Filter Line-Search Algorithm
  for Large-Scale Nonlinear Programming*, Mathematical Programming 106
  (2006), 25--57, DOI 10.1007/s10107-004-0559-y.
- **MLALM** follows Q. Shi, X. Wang, and H. Wang, *A Momentum-Based
  Linearized Augmented Lagrangian Method for Nonconvex Constrained Stochastic
  Optimization*, Mathematics of Operations Research 51 (2026), 92--133,
  DOI 10.1287/moor.2022.0193.
- **S-SQP** follows A. S. Berahas, F. E. Curtis, D. Robinson, and B. Zhou,
  *Sequential Quadratic Optimization for Nonlinear Equality Constrained
  Stochastic Optimization*, SIAM Journal on Optimization 31 (2021),
  1352--1379, DOI 10.1137/20M1354556.

Algorithm labels, stopping criteria, and work accounting are frozen in the
section-specific configuration files. Please cite the original method papers
when using or modifying these implementations.
