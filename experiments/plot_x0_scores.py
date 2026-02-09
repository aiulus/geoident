"""Offline utility to create boxplots from pre-computed x0 scores.

This script loads a pbh_scores.csv file and creates boxplots without
needing to recompute anything.

Example
-------
python -m pyident.experiments.plot_x0_scores \\
    --scores-csv pyident_results/x0_boxplot/pbh_scores.csv \\
    --outdir pyident_results/x0_boxplot
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

import numpy as np
import pandas as pd


def format_density_label(density: float) -> str:
    """Format density value for display."""
    if density >= 1.0:
        return "ρ=1"
    return f"ρ={density:g}"


def run(args: argparse.Namespace) -> None:
    # Load scores
    scores_df = pd.read_csv(args.scores_csv)
    
    if scores_df.empty:
        raise RuntimeError(f"No data in {args.scores_csv}")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)

    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting; please install it to run this experiment"
        ) from _MATPLOTLIB_IMPORT_ERROR

    # Get unique x0 densities in sorted order
    if "x0_density" in scores_df.columns:
        x0_densities = sorted(scores_df["x0_density"].unique())
        density_col = "x0_density"
        label_func = format_density_label
    elif "p_keep" in scores_df.columns:
        # Fallback for sim_unctrb_x0_boxplot format
        x0_densities = sorted(scores_df["p_keep"].unique())
        density_col = "p_keep"
        label_func = lambda p: "p=1" if p >= 1.0 else f"p={p:g}"
    else:
        raise ValueError("Could not find x0_density or p_keep column in CSV")

    # Prepare data for plot
    score_col = args.score_column
    if score_col not in scores_df.columns:
        available = [col for col in scores_df.columns if col not in ["n", "m", "system_index"]]
        raise ValueError(f"Column '{score_col}' not found. Available: {available}")

    # Binarize scores if requested
    if args.binary:
        plot_data = scores_df[score_col] > args.epsilon
        plot_data = plot_data.astype(int)
        plot_col_name = f"identifiable (ε={args.epsilon})"
    else:
        plot_data = scores_df[score_col]
        plot_col_name = f"{score_col} score"

    # Add plot data to dataframe
    scores_df["_plot_data"] = plot_data

    data_by_density = []
    labels_by_density = []
    for x0_density in x0_densities:
        sub = scores_df[scores_df[density_col] == x0_density]
        vals = sub["_plot_data"].to_numpy()
        vals = vals[np.isfinite(vals)]  # Remove NaN/Inf
        data_by_density.append(vals)
        labels_by_density.append(label_func(x0_density))

    # Create appropriate plot
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if args.binary:
        # For binary data, show fraction identifiable as bar plot
        fractions = [np.mean(data) for data in data_by_density]
        counts = [np.sum(data).astype(int) for data in data_by_density]
        totals = [len(data) for data in data_by_density]
        
        bars = ax.bar(labels_by_density, fractions, width=0.6, color=(32/255, 143/255, 140/255), alpha=0.8, edgecolor="black")
        
        # Add count labels on bars
        for bar, count, total in zip(bars, counts, totals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}/{total}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("Fraction Identifiable", fontsize=18)
        ax.set_title(
            f"Fraction Identifiable ({score_col} > {args.epsilon}) \n vs x0 Density Level",
            fontsize=20,
            fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
    else:
        # For continuous data, show boxplot
        bp = ax.boxplot(
            data_by_density,
            labels=labels_by_density,
            patch_artist=True,
            widths=0.6,
        )

        # Style boxes
        for patch in bp["boxes"]:
            patch.set_facecolor((32/255, 143/255, 140/255))
            patch.set_alpha(0.8)

        ax.set_ylabel(f"{score_col} Score", fontsize=18)
        ax.set_title(
            f"{score_col.upper()} Score Distribution vs x0 Density Level",
            fontsize=20,
            fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

        if args.yscale:
            ax.set_yscale(args.yscale)

    ax.set_xlabel("x0 Sampling Scheme", fontsize=18)

    fig.tight_layout()
    plot_kind = "binary" if args.binary else "boxplot"
    plot_path = outdir / "plots" / f"{score_col}_{plot_kind}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Plot saved to {plot_path}")
    print(f"  Score column: {score_col}")
    print(f"  x0 densities: {x0_densities}")
    if args.binary:
        print(f"  Binary threshold (ε): {args.epsilon}")
        for density, label, data in zip(x0_densities, labels_by_density, data_by_density):
            frac = np.mean(data)
            count = int(np.sum(data))
            total = len(data)
            print(f"  {label:10s}: {count:4d}/{total:4d} identifiable ({frac:.2%})")
    else:
        for density, label, data in zip(x0_densities, labels_by_density, data_by_density):
            print(f"  {label:10s}: n={len(data):4d}, median={np.median(data):.4f}, mean={np.mean(data):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--scores-csv",
        required=True,
        help="Path to pbh_scores.csv (or other scores CSV with x0_density column)",
    )

    parser.add_argument(
        "--outdir",
        default="results/x0_plot",
        help="Output directory for plots",
    )

    parser.add_argument(
        "--score-column",
        default="pbh",
        help="Column name to plot (e.g., 'pbh', 'value', 'mu')",
    )

    parser.add_argument(
        "--yscale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale",
    )

    parser.add_argument(
        "--binary",
        action="store_true",
        help="Binarize scores as identifiable = 1 iff score > epsilon, else 0",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Threshold for identifiability when --binary is used (default: 1e-6)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
