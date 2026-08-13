# NR-LALM: theory formalization and numerical experiments

This repository is the computational companion to the paper
*A Fixed-Penalty Linearized Augmented Lagrangian Method with Classical
Multiplier Updates*. It provides the numerical implementation and complete
reproduction workflow for Sections 5.1--5.3. The paper's core theoretical
results are independently formalized and machine-checked in Lean 4 in the
[ReasBook formalization][lean-readme].

Together, the two resources provide complementary verification:

- **machine-checkable theory:** a version-pinned Lean 4 development covering
  the principal deterministic, stochastic, Kurdyka--Lojasiewicz (KL), and
  second-order-correction (SOC) results; and
- **reproducible computation:** algorithm implementations, frozen paper
  configurations, public-data preparation, experiment runners, common residual
  evaluators, analysis and rendering code, and regression tests.

## Resources

| Resource | Purpose |
|---|---|
| [Lean source at ReasBook v4.32.2][lean-source] | Version-pinned formal source for the paper |
| [Formalization README][lean-readme] | Scope, verification commands, source layout, and article-to-Lean correspondence |
| [Interactive theorem-dependency map][theorem-map] | Searchable graph of the 24 article-level entries and their dependencies |
| [Aggregate Lean module][lean-aggregate] | Compact import surface for the complete article-facing development |
| [Paper module][lean-paper] | ReasBook documentation wrapper for the paper |
| [Numerical reproducibility guide](docs/REPRODUCIBILITY.md) | Commands and paper-artifact mapping for Sections 5.1--5.3 |
| [Baseline documentation](docs/BASELINES.md) | Sources and scope of the independent baseline implementations |

All formalization links are pinned to ReasBook `v4.32.2`, the version used by
the manuscript. The numerical repository contains source and generic run
tooling; generated data, numerical records, figures, tables, and cluster logs
are intentionally not versioned. The commands below recreate the paper
artifacts in the corresponding `results/` directories.

## Theory formalization

The Lean development models the paper's mathematical objects and proves its
theoretical guarantees. Its scope includes:

- the local smoothness and uniform linear independence constraint
  qualification assumptions and approximate Karush--Kuhn--Tucker points;
- the fixed-penalty NR-LALM iteration with the classical nonlinear-residual
  multiplier update;
- parameter existence, step--multiplier invariants, Lyapunov descent, and
  deterministic trajectory localization;
- deterministic iteration and oracle complexity of order
  `O(epsilon^-2)`;
- finite-length primal--dual convergence under a KL condition;
- stochastic-oracle and projected stochastic path-integrated differential
  estimator models, including localization and safeguarded restart;
- stochastic-gradient complexity of order `O(epsilon^-3)` and deterministic
  evaluation complexity of order `O(epsilon^-2)`; and
- the optional minimum-norm SOC and its sufficient-region comparison with
  NR-LALM.

The formalization is proof-oriented: its structures encode the iterations and
invariants needed for theorem proving and are not the executable numerical
implementation. Compound statements from the paper may therefore correspond
to several focused Lean declarations. The [formalization README][lean-readme]
gives the complete article-to-Lean table, and the [interactive theorem map]
[theorem-map] gives a convenient graphical entry point.

To check the development, clone
[ReasBook](https://github.com/optpku/ReasBook), select the version used by the
paper, and run the article aggregate from the ReasBook project environment:

```bash
git clone https://github.com/optpku/ReasBook.git
cd ReasBook
git checkout v4.32.2
cd ReasBook
lake lean Papers/TR_LALM_theory.lean
lake lean Papers/TR_LALM_theory/Paper.lean
```

Lean contributor: [Zichen Wang](https://github.com/imathwy).

## Numerical repository map

| Paper part | Directory | Paper-facing output |
|---|---|---|
| Section 5.1 | `experiments/section_5_1_mechanism` | mechanism-verification figure |
| Section 5.2 | `experiments/section_5_2_deterministic` | 15-data-set timing table |
| Section 5.3 | `experiments/section_5_3_stochastic` | two-panel mean squared KKT figure |

The paper names are used throughout: **NR-LALM** is the base method and
**NR-LALM+SOC** is its second-order-correction variant.

## Numerical installation

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest
python scripts/validate_release.py
```

IPOPT is used only in Section 5.2. Reproducing the complete four-method table
also requires a system IPOPT installation with MUMPS and a compatible
`cyipopt` build. The other experiments and all non-IPOPT unit tests do not
require it.

## Quick numerical reproduction

Section 5.1 is self-contained:

```bash
python experiments/section_5_1_mechanism/scripts/run_mechanism_verification.py \
  --config experiments/section_5_1_mechanism/configs/paper_v1.toml \
  --output-dir experiments/section_5_1_mechanism/results/paper_run
```

Sections 5.2 and 5.3 use public LIBSVM data and are intended for a CPU
cluster. Their own READMEs give the exact download, array-job, aggregation,
and rendering commands. Section 5.2 downloads several very large official
archives before forming deterministic 20,000-row samples; check storage and
network quotas first.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete mapping
from commands to paper artifacts.

## Citation

If you use the algorithms, theoretical formalization, or numerical software,
please cite the accompanying paper. Until the arXiv record is available, use
the following preprint entry:

```bibtex
@misc{LiuDengWangWen2026NRLALM,
  author = {Benqi Liu and Kangkang Deng and Zichen Wang and Zaiwen Wen},
  title  = {A Fixed-Penalty Linearized Augmented Lagrangian Method with
            Classical Multiplier Updates},
  year   = {2026},
  note   = {Preprint}
}
```

The arXiv identifier and URL will be added after the preprint is posted.
Machine-readable citation metadata is provided in
[`CITATION.cff`](CITATION.cff).

## Authors and contact

- Benqi Liu: [bqliu@pku.edu.cn](mailto:bqliu@pku.edu.cn)
- Kangkang Deng: [freedeng1208@gmail.com](mailto:freedeng1208@gmail.com)
- Zichen Wang: [zichenwang25@stu.pku.edu.cn](mailto:zichenwang25@stu.pku.edu.cn)
- Zaiwen Wen (corresponding author): [wenzw@pku.edu.cn](mailto:wenzw@pku.edu.cn)

## License

Code is provided under the MIT License. Baseline implementations are
independent equation-level implementations; their sources are cited in
[docs/BASELINES.md](docs/BASELINES.md).

[lean-source]: https://github.com/optpku/ReasBook/tree/v4.32.2/ReasBook/Papers/TR_LALM_theory/
[lean-readme]: https://github.com/optpku/ReasBook/blob/v4.32.2/ReasBook/Papers/TR_LALM_theory/README.md
[theorem-map]: https://optpku.github.io/ReasBook/theorem-maps/papers/tr_lalm_theory/
[lean-aggregate]: https://github.com/optpku/ReasBook/blob/v4.32.2/ReasBook/Papers/TR_LALM_theory.lean
[lean-paper]: https://github.com/optpku/ReasBook/blob/v4.32.2/ReasBook/Papers/TR_LALM_theory/Paper.lean
