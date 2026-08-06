# Section 5.1: mechanism verification

This self-contained experiment generates the two-panel mechanism figure. The
left panel evaluates the sufficient parameter inequalities for NR-LALM and
NR-LALM+SOC on a fixed 900-by-1000 grid. The right panel fits the base and SOC
constraint-linearization errors along 32 fixed directions.

Run from the repository root:

```bash
python experiments/section_5_1_mechanism/scripts/run_mechanism_verification.py \
  --config experiments/section_5_1_mechanism/configs/paper_v1.toml \
  --output-dir experiments/section_5_1_mechanism/results/paper_run
```

The paper-facing file is `mechanism_verification.pdf`;
`figure_caption.tex` contains its caption, while `raw.json` stores the
computed grid counts and fitted orders.

This figure illustrates sufficient analytical regions and local error
orders. It is not an empirical convergence-basin or runtime comparison.
