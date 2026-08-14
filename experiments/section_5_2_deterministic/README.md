# Section 5.2: deterministic high-dimensional comparison

This experiment reproduces the 15-row timing table for NR-LALM,
NR-LALM+SOC, strict L-AL, and IPOPT.

## 1. Prepare public data

The command downloads the official source archives and derives the 15 paper
data sets:

```bash
python experiments/section_5_2_deterministic/scripts/prepare_data.py
```

Some official archives are very large even though the deterministic output
is capped at 20,000 rows. To prepare one case first, pass its data-set index,
for example `--dataset-index 3` for duke breast-cancer.

## 2. Run the balanced timing experiment

The intended execution is one exclusive CPU allocation per data set:

```bash
cd experiments/section_5_2_deterministic
sbatch scripts/slurm_array.sbatch
```

Set `PYTHON_EXE`, `CONFIG`, or `RESULTS_ROOT` in the environment to override
the defaults. A sequential local launcher is also provided:

```bash
python experiments/section_5_2_deterministic/scripts/run_all_datasets.py
```

Each method process has a default 1,800-second timeout. Thread counts for
BLAS, OpenMP, MKL, NumExpr, and Accelerate are fixed to one.

## 3. Analyze and render the table

```bash
python experiments/section_5_2_deterministic/scripts/analyze_results.py \
  --config experiments/section_5_2_deterministic/configs/paper_benchmark.json \
  --results-root experiments/section_5_2_deterministic/results/paper_run \
  --output-json experiments/section_5_2_deterministic/results/paper_run/summary.json \
  --output-csv experiments/section_5_2_deterministic/results/paper_run/timings.csv

python experiments/section_5_2_deterministic/scripts/render_table.py \
  --summary experiments/section_5_2_deterministic/results/paper_run/summary.json \
  --output-dir experiments/section_5_2_deterministic/results/paper_run/table \
  --table-only
```

The final LaTeX artifact is `libsvm_median_timing_table.tex`. The
analyzer rejects incomplete task sets, inconsistent traces, and altered
residual identities.

The paper configuration uses IPOPT 3.14.19 with `linear_solver=pardisomkl`,
linked to Intel oneAPI Math Kernel Library 2025.3.0 through `cyipopt` 1.6.1.
The other three methods use the included matrix-free/Woodbury implementation.
