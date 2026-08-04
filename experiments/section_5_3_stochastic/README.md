# Section 5.3: stochastic KKT comparison

This experiment compares NR-LALM, NR-LALM+SOC, MLALM, and S-SQP on the
public covtype and MNIST multiclass data sets. It reports the unsmoothed
pointwise arithmetic mean of a common full-data squared KKT residual over ten
independent stochastic-oracle streams per data set.

## 1. Download data

```bash
python experiments/section_5_3_stochastic/scripts/download_data.py
```

## 2. Run 20 data-set/stream tasks

On Slurm:

```bash
cd experiments/section_5_3_stochastic
sbatch scripts/slurm_array.sbatch
```

The array range is 0--19. Each task runs all four methods and produces four
records, for 80 method runs in total. The launcher is CPU-only; no GPU is
used. A sequential local end-to-end command is available, but is slower:

```bash
python experiments/section_5_3_stochastic/scripts/run_all.py
```

## 3. Aggregate and render after an array run

```bash
python experiments/section_5_3_stochastic/scripts/analyze_results.py \
  --config experiments/section_5_3_stochastic/configs/paper_v1.json \
  --raw-dir experiments/section_5_3_stochastic/results/paper_run/raw \
  --output-dir experiments/section_5_3_stochastic/results/paper_run/analysis

python experiments/section_5_3_stochastic/scripts/render_figure.py \
  --config experiments/section_5_3_stochastic/configs/paper_v1.json \
  --analysis-dir experiments/section_5_3_stochastic/results/paper_run/analysis \
  --output-dir experiments/section_5_3_stochastic/results/paper_run/figure
```

The paper-facing artifact is `stochastic_kkt_residual_two_panel.pdf`.
Analysis requires all 80 completed records and exact method/data/stream task
identities before it writes a curve.
