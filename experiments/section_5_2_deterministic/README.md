# Section 5.2: deterministic high-dimensional comparison

This experiment reproduces the 15-row timing table for NR-LALM,
NR-LALM+SOC, strict L-AL, and IPOPT.

## 1. Prepare public data

The full command downloads the official source archives and derives only the
15 predeclared high-dimensional cases:

```bash
python experiments/section_5_2_deterministic/scripts/prepare_stage_b_data.py
```

Some official archives are very large even though the deterministic output
is capped at 20,000 rows. To prepare one case first, pass its Stage-B index,
for example `--dataset-index 3` for duke breast-cancer.

## 2. Run the balanced timing experiment

The intended execution is one exclusive CPU allocation per data set:

```bash
cd experiments/section_5_2_deterministic
sbatch scripts/slurm_stage_b.sbatch
```

Set `PYTHON_EXE`, `CONFIG`, or `RESULTS_ROOT` in the environment to override
the defaults. A sequential local launcher is also provided:

```bash
python experiments/section_5_2_deterministic/scripts/run_all_datasets.py
```

Each method process has a default 1,800-second timeout. Thread counts for
BLAS, OpenMP, MKL, NumExpr, and Accelerate are fixed to one.

## 3. Audit and render the table

```bash
python experiments/section_5_2_deterministic/scripts/analyze_libsvm_suite_stage_b.py \
  --config experiments/section_5_2_deterministic/configs/paper_stage_b_v2.json \
  --results-root experiments/section_5_2_deterministic/results/paper_stage_b \
  --output-json experiments/section_5_2_deterministic/results/paper_stage_b/summary.json \
  --output-csv experiments/section_5_2_deterministic/results/paper_stage_b/timings.csv

python experiments/section_5_2_deterministic/scripts/render_libsvm_suite_stage_b_outputs.py \
  --summary experiments/section_5_2_deterministic/results/paper_stage_b/summary.json \
  --output-dir experiments/section_5_2_deterministic/results/paper_stage_b/table \
  --table-only
```

The final LaTeX artifact is `libsvm_stage_b_median_timing_v4.tex`. The
analyzer refuses to pass incomplete task sets, invalid checksums, inconsistent
traces, or altered residual identities.

IPOPT requires a working MUMPS-backed IPOPT and `cyipopt`; the other three
methods use the included matrix-free/Woodbury implementation.
