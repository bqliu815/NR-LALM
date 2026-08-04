# Reproducibility map

## Section 5.1: mechanism verification

Input: `experiments/section_5_1_mechanism/configs/paper_v1.toml`.

Runner: `experiments/section_5_1_mechanism/scripts/run_mechanism_verification.py`.

Paper-facing output: `mechanism_verification.pdf`. The same run also writes a
300-dpi PNG, the LaTeX caption, numerical grid counts, directional slopes,
and a machine-readable validation record.

## Section 5.2: deterministic public-data comparison

Inputs: the complete LIBSVM manifest and
`experiments/section_5_2_deterministic/configs/paper_stage_b_v2.json`.

Pipeline:

1. `prepare_stage_b_data.py` downloads and deterministically derives the 15
   high-dimensional data files.
2. `run_stage_b_dataset.py` launches four fresh method processes under eight
   balanced orders for one data set. `slurm_stage_b.sbatch` maps the 15 data
   sets to an array job.
3. `analyze_libsvm_suite_stage_b.py` validates all 480 method runs and writes
   the audited timing summary.
4. `render_libsvm_suite_stage_b_outputs.py --table-only` writes the LaTeX
   timing table used in the paper.

First-hit time includes the common independent KKT evaluator but excludes
file parsing and instance construction, exactly as stated in the paper.

## Section 5.3: stochastic public-data comparison

Input: `experiments/section_5_3_stochastic/configs/paper_v1.json`.

Pipeline:

1. `download_data.py` downloads and verifies covtype and MNIST.
2. `run_repeat.py` runs NR-LALM, NR-LALM+SOC, MLALM, and S-SQP for one
   data-set/stream pair. `slurm_array.sbatch` supplies the 20 array indices.
3. `analyze_results.py` requires all 80 method records, applies common
   full-data KKT checks, and computes pointwise arithmetic means.
4. `render_figure.py` writes `stochastic_kkt_residual_two_panel.pdf` using the
   Section 5.1 physical size, typography, colors, and white background.

The horizontal axis counts sampled objective component-gradient evaluations
after the common warm start. The vertical metric is the squared KKT residual
computed for every method with the same full-data least-squares multiplier.
No smoothing, running best, uncertainty band, or selective checkpointing is
used.
