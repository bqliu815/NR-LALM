# NR-LALM numerical experiments

This repository contains the code needed to reproduce Sections 5.1--5.3 of
the paper *A Fixed-Penalty Linearized Augmented Lagrangian Method with
Classical Multiplier Updates*.

The repository contains the algorithm implementations, paper configurations,
public-data preparation code, experiment runners, analysis and rendering code,
and regression tests. Generated data, numerical records, figures, tables, and
cluster logs are intentionally not versioned; the commands below recreate the
paper artifacts in the corresponding `results/` directories.

## Repository map

| Paper part | Directory | Paper-facing output |
|---|---|---|
| Section 5.1 | `experiments/section_5_1_mechanism` | mechanism-verification figure |
| Section 5.2 | `experiments/section_5_2_deterministic` | 15-data-set timing table |
| Section 5.3 | `experiments/section_5_3_stochastic` | two-panel mean squared KKT figure |

The paper names are used throughout: **NR-LALM** is the base method and
**NR-LALM+SOC** is its second-order-correction variant.

## Installation

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

## Quick reproduction

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

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the mapping from
commands to paper artifacts.

## License and citation

Code is provided under the MIT License. Baseline implementations are
independent equation-level implementations; their sources are cited in
[docs/BASELINES.md](docs/BASELINES.md). Please cite the accompanying paper
and this software using `CITATION.cff`.
