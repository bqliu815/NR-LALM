#!/usr/bin/env python3
"""Render the Section 5.2 LIBSVM table and optional summary figure."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ["nr_lalm", "nr_lalm_soc", "l_al", "ipopt"]
METHOD_LABELS = {
    "nr_lalm": "NR-LALM",
    "nr_lalm_soc": "NR-LALM+SOC",
    "l_al": "L-AL",
    "ipopt": "IPOPT",
}
METHOD_COLORS = {
    "nr_lalm": "#0072B2",
    "nr_lalm_soc": "#D55E00",
    "l_al": "#009E73",
    "ipopt": "#6A3D9A",
}
METHOD_MARKERS = {
    "nr_lalm": "o",
    "nr_lalm_soc": "s",
    "l_al": "^",
    "ipopt": "D",
}
METHOD_LINESTYLES = {
    "nr_lalm": "-",
    "nr_lalm_soc": "-",
    "l_al": "--",
    "ipopt": "-.",
}
METHOD_HATCHES = {
    "nr_lalm": "",
    "nr_lalm_soc": "//",
    "l_al": "..",
    "ipopt": "xx",
}
SHORT_DATASET_LABELS = {
    "avazu-app": "avazu-app",
    "avazu-site": "avazu-site",
    "criteo": "criteo",
    "duke-breast-cancer": "duke breast-cancer",
    "gisette": "gisette",
    "kdd2010-algebra": "kdd2010 (algebra)",
    "kdd2010-bridge-to-algebra": "kdd2010 (bridge-to-algebra)",
    "kdd2010-raw-version-bridge-to-algebra": "kdd2010 raw (bridge-to-algebra)",
    "kdd2012": "kdd2012",
    "leukemia": "leukemia",
    "news20-binary": "news20.binary",
    "rcv1-binary": "rcv1.binary",
    "real-sim": "real-sim",
    "url": "url",
    "webspam": "webspam",
}

# Numbers of observations in the classwise rounded 80% training portions.
TRAINING_SAMPLES = {
    "avazu-app": 16000,
    "avazu-site": 16000,
    "criteo": 16000,
    "duke-breast-cancer": 35,
    "gisette": 4800,
    "kdd2010-algebra": 16000,
    "kdd2010-bridge-to-algebra": 16000,
    "kdd2010-raw-version-bridge-to-algebra": 16000,
    "kdd2012": 16000,
    "leukemia": 31,
    "news20-binary": 15997,
    "rcv1-binary": 16000,
    "real-sim": 16000,
    "url": 16000,
    "webspam": 16000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_key(threshold: float) -> str:
    exponent = -int(f"{threshold:.0e}".split("e")[1])
    return f"r2_1e_minus_{exponent}"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def format_seconds(value: float) -> str:
    if value < 0.01:
        return f"{value:.3f}"
    if value < 10.0:
        return f"{value:.2f}"
    if value < 1000.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def format_paper_seconds(value: float) -> str:
    """Format the median-only paper table without hiding sub-0.1s gaps."""
    if value < 0.1:
        return f"{value:.3f}"
    if value < 10.0:
        return f"{value:.2f}"
    if value < 1000.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def timing_cell(
    row: dict[str, Any],
    method: str,
    timeout_count: int,
    expected_orders: int,
) -> str:
    successes = int(row[f"{method}_successes"])
    if successes == 0:
        if timeout_count == expected_orders:
            return r"\mathrm{TO}"
        return rf"--\ ({successes}/{expected_orders})"
    median = float(row[f"{method}_median_seconds"])
    q1 = float(row[f"{method}_r2_1e_minus_8_q1"])
    q3 = float(row[f"{method}_r2_1e_minus_8_q3"])
    value = (
        f"{format_seconds(median)} "
        f"[{format_seconds(q1)}, {format_seconds(q3)}]"
    )
    if successes < expected_orders:
        value += f" ({successes}/{expected_orders})"
    return value


def successful_dataset_counts(
    timing_rows: list[dict[str, Any]],
    methods: list[str],
    thresholds: list[float],
    expected_orders: int,
) -> dict[str, list[int]]:
    return {
        method: [
            sum(
                int(
                    row[
                        f"{method}_{threshold_key(threshold)}_count"
                    ]
                )
                == expected_orders
                for row in timing_rows
            )
            for threshold in thresholds
        ]
        for method in methods
    }


def configure_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "lines.linewidth": 1.2,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.8,
        }
    )


def render_figure(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    timing_rows = list(summary["timing_rows"])
    expected_orders = int(summary["expected_orders"])
    thresholds = [
        float(summary["primary_residual_squared_threshold"]),
        *[
            float(value)
            for value in summary[
                "secondary_residual_squared_thresholds"
            ]
        ],
    ]
    complete_counts = successful_dataset_counts(
        timing_rows, METHODS, thresholds, expected_orders
    )
    configure_style(plt)
    figure, axes = plt.subplots(
        1, 2, figsize=(5.125, 2.45), facecolor="white"
    )

    axis = axes[0]
    tau_values = np.asarray(
        summary["performance_profile_tau_grid"], dtype=np.float64
    )
    for method in METHODS:
        profile = summary["performance_profile"][method]
        values = np.asarray(
            [float(profile[f"{tau:.12g}"]) for tau in tau_values],
            dtype=np.float64,
        )
        marker_indices = np.linspace(
            0, len(tau_values) - 1, num=7, dtype=int
        )
        axis.semilogx(
            tau_values,
            values,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markevery=marker_indices,
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=METHOD_LABELS[method],
        )
    axis.set_xlim(1.0, 100.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel(r"performance ratio $\tau$")
    axis.set_ylabel("fraction of data sets")
    axis.set_title("(a) Time performance profile")
    legend = axis.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        borderpad=0.25,
    )
    legend.get_frame().set_linewidth(1.0)
    axis.grid(which="both", color="#E0E0E0", linewidth=1.0)

    axis = axes[1]
    x_values = np.arange(len(thresholds), dtype=np.float64)
    width = 0.19
    offsets = (np.arange(len(METHODS)) - 1.5) * width
    for method, offset in zip(METHODS, offsets):
        bars = axis.bar(
            x_values + offset,
            complete_counts[method],
            width=width,
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            hatch=METHOD_HATCHES[method],
            alpha=0.88,
        )
        axis.bar_label(
            bars,
            labels=[str(value) for value in complete_counts[method]],
            padding=1.2,
            fontsize=6.8,
        )
    axis.set_xticks(
        x_values,
        [
            rf"$10^{{-{int(-math.log10(threshold))}}}$"
            for threshold in thresholds
        ],
    )
    axis.set_ylim(0.0, len(timing_rows) + 1.0)
    axis.set_yticks(np.arange(0, len(timing_rows) + 1, 3))
    axis.set_xlabel(r"squared-residual threshold")
    axis.set_ylabel("data sets solved in 8/8 runs")
    axis.set_title("(b) All-repeat successes")
    axis.grid(axis="y", color="#E0E0E0", linewidth=1.0)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout(w_pad=1.2)

    pdf_path = output_dir / "libsvm_stage_b_performance_v1.pdf"
    png_path = output_dir / "libsvm_stage_b_performance_v1.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(
        png_path, dpi=300, bbox_inches="tight", pad_inches=0.08
    )
    plt.close(figure)
    return [pdf_path, png_path]


def render_table(summary: dict[str, Any], output_dir: Path) -> Path:
    timing_rows = list(summary["timing_rows"])
    expected_orders = int(summary["expected_orders"])
    timeout_counts: Counter[tuple[str, str]] = Counter(
        (
            str(record["dataset"]),
            str(record["method"]),
        )
        for record in summary["raw_records"]
        if record["audit_status"] == "external_timeout_1800s"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Median first-hit time (seconds) at "
        r"$\mathcal R_k^2\le10^{-8}$ over eight balanced runs.  Bold marks "
        r"the fastest successful method in each row; $\mathrm{TO}$ denotes "
        r"eight external timeouts.}",
        r"\label{tab:libsvm-stage-b-timing}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Data set & $N$ & $n$ & NR-LALM & NR-LALM+SOC & L-AL & IPOPT \\",
        r"\midrule",
    ]
    for row in timing_rows:
        dataset = str(row["dataset"])
        complete_methods = [
            method
            for method in METHODS
            if int(row[f"{method}_successes"]) == expected_orders
        ]
        fastest = math.inf
        if complete_methods:
            fastest = min(
                float(row[f"{method}_median_seconds"])
                for method in complete_methods
            )
        cells = []
        for method in METHODS:
            successes = int(row[f"{method}_successes"])
            if successes == 0:
                if timeout_counts[(dataset, method)] == expected_orders:
                    cell = r"\(\mathrm{TO}\)"
                else:
                    cell = "--"
            else:
                cell = format_paper_seconds(
                    float(row[f"{method}_median_seconds"])
                )
            median = row[f"{method}_median_seconds"]
            if (
                method in complete_methods
                and median is not None
                and math.isclose(
                    float(median), fastest, rel_tol=1.0e-12, abs_tol=0.0
                )
            ):
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        dataset_label = SHORT_DATASET_LABELS.get(
            dataset, str(row["display_name"])
        )
        dimension = f"{int(row['dimension']):,}".replace(",", r"{,}")
        training_samples = f"{TRAINING_SAMPLES[dataset]:,}".replace(
            ",", r"{,}"
        )
        lines.append(
            " & ".join(
                [
                    latex_escape(dataset_label),
                    f"${training_samples}$",
                    f"${dimension}$",
                    *cells,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    table_path = output_dir / "libsvm_stage_b_median_timing_v4.tex"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="write the LaTeX table without importing Matplotlib",
    )
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("passed", False):
        raise ValueError("refusing to render an unpassed Stage-B audit")
    if (
        int(summary["expected_datasets"]) != 15
        or int(summary["expected_orders"]) != 8
        or int(summary["expected_methods"]) != 4
    ):
        raise ValueError("unexpected Stage-B experiment shape")
    if set(summary["method_summaries"]) != set(METHODS):
        raise ValueError("unexpected method set")

    outputs: list[Path] = []
    if not args.table_only:
        outputs.extend(render_figure(summary, output_dir))
    outputs.append(render_table(summary, output_dir))
    if not args.table_only:
        caption_path = output_dir / "libsvm_stage_b_figure_caption.tex"
        caption_path.write_text(
            (
                r"\caption{Complete high-dimensional LIBSVM comparison. "
                r"\textup{(a)} Performance profiles for median first-hit "
                r"time at $\mathcal R_k^2\le10^{-8}$, with failures "
                r"assigned infinite ratio. \textup{(b)} Data sets for "
                r"which all eight runs reached each squared-residual "
                r"threshold.}"
                "\n"
            ),
            encoding="utf-8",
        )
        outputs.append(caption_path)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": "libsvm_stage_b_paper_outputs_v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_summary_path": str(summary_path),
        "input_summary_sha256": sha256(summary_path),
        "input_audit_passed": True,
        "expected_datasets": 15,
        "expected_orders": 8,
        "methods": METHODS,
        "primary_residual_squared_threshold": 1.0e-8,
        "table_policy": (
            "median first-hit seconds; failures remain in the denominator; "
            "bold requires 8/8 success and minimum median time"
        ),
        "render_mode": "table_only" if args.table_only else "full",
        "training_samples": TRAINING_SAMPLES,
        "figure_size_inches": (
            None if args.table_only else [5.125, 2.45]
        ),
        "outputs": {
            path.name: sha256(path)
            for path in outputs
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "outputs": [str(path) for path in outputs],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
