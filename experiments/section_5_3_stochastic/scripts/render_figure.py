#!/usr/bin/env python3
"""Render the Section 5.3 two-panel KKT figure in the paper style."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402


METHODS = ("NR-LALM", "NR-LALM+SOC", "MLALM", "S-SQP")
COLORS = {
    "NR-LALM": "#0072B2",
    "NR-LALM+SOC": "#D55E00",
    "MLALM": "#009E73",
    "S-SQP": "#6A3D9A",
}
MARKERS = {"NR-LALM": "o", "NR-LALM+SOC": "s", "MLALM": "^", "S-SQP": "D"}
LINESTYLES = {
    "NR-LALM": "-",
    "NR-LALM+SOC": (0, (2.2, 1.2)),
    "MLALM": "--",
    "S-SQP": "-.",
}
MARK_EVERY = {"NR-LALM": (0, 2), "NR-LALM+SOC": (1, 2), "MLALM": 1, "S-SQP": 1}
DISPLAY_NAMES = {"covtype": "covtype", "mnist": "MNIST"}
FIGURE_SIZE = (5.1130, 2.4678)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "lines.linewidth": 1.2,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def selected(rows: list[dict[str, str]], dataset: str, method: str) -> list[dict[str, str]]:
    return sorted(
        [row for row in rows if row["dataset"] == dataset and row["method"] == method],
        key=lambda row: int(row["checkpoint"]),
    )


def finish_axis(axis: plt.Axes) -> None:
    axis.set_xlabel("Stochastic-gradient evaluations")
    axis.set_ylabel("Mean squared KKT residual")
    axis.set_xlim(left=0.0)
    axis.set_yscale("log")
    curve_maximum = max(float(np.max(line.get_ydata())) for line in axis.lines)
    axis.set_ylim(top=6.0 * curve_maximum)
    axis.yaxis.set_minor_locator(mticker.NullLocator())
    axis.grid(which="major", color="#E0E0E0", linewidth=1.0)
    axis.grid(False, which="minor")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((5, 5))
    axis.xaxis.set_major_formatter(formatter)
    handles, labels = axis.get_legend_handles_labels()
    order = (0, 2, 1, 3)
    legend = axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        borderpad=0.25,
        handlelength=1.65,
        handletextpad=0.38,
        labelspacing=0.22,
        columnspacing=0.75,
    )
    legend.get_frame().set_linewidth(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = json.loads((args.analysis_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or tuple(config["displayed_methods"]) != METHODS:
        raise RuntimeError("validated four-method results are required")
    with (args.analysis_dir / "curves.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, facecolor="white")
    for panel, (axis, dataset) in enumerate(zip(axes, ("covtype", "mnist"))):
        for method in METHODS:
            method_rows = selected(rows, dataset, method)
            if len(method_rows) != int(config["curve_points"]):
                raise RuntimeError(f"wrong curve length for {dataset}/{method}")
            axis.plot(
                [float(row["component_calls"]) for row in method_rows],
                [float(row["optimized_pair_residual_sq_mean"]) for row in method_rows],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker=MARKERS[method],
                markerfacecolor="none",
                markeredgewidth=1.0,
                markersize=4.5,
                linewidth=1.2,
                markevery=MARK_EVERY[method],
                label=method,
            )
        axis.set_title(f"({chr(ord('a') + panel)}) {DISPLAY_NAMES[dataset]}")
        finish_axis(axis)
    figure.tight_layout(w_pad=1.2)
    figure.savefig(
        args.output_dir / "stochastic_kkt_residual_two_panel.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    figure.savefig(
        args.output_dir / "stochastic_kkt_residual_two_panel.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.085,
    )
    plt.close(figure)
    repeats = int(config["oracle_repeats"])
    repeat_label = "ten" if repeats == 10 else str(repeats)
    caption = (
        r"\caption{Mean squared least-squares-multiplier KKT residual versus "
        r"stochastic-gradient evaluations on \textup{(a)} covtype and "
        r"\textup{(b)} MNIST. Curves are arithmetic means over "
        + repeat_label
        + r" independent stochastic-oracle streams; the KKT residual of every "
        r"method uses the same full-data least-squares multiplier.}"
    )
    (args.output_dir / "figure_caption.tex").write_text(caption + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
