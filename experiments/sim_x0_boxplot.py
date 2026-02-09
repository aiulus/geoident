"""Box plots of PBH scores for different x0 sampling schemes.

Workflow
--------
1) Load (A,B) ensemble from sim_regcomb_ctrb.py output
2) Select systems in a given state dimension and density range
3) Sample x0 via different sparsification levels (x0_density in [0,1])
   - x0_density=1: uniform sampling from unit sphere
   - x0_density<1: uniform sphere + Bernoulli sparsification (keep with prob x0_density) + renormalize
4) Compute PBH score for each (A,B,x0) triple
5) Create boxplot showing distributions per x0_density

Example
-------
python -m pyident.experiments.sim_x0_boxplot \\
    --dataset-csv pyident_results/iclr_manuscript/ensemble/systems_unctrb_d0.3_0.7.csv \\
    --dataset-npz pyident_results/iclr_manuscript/ensemble/systems_unctrb_d0.3_0.7.npz \\
    --x0-densities 0.25 0.5 0.75 1.0 \\
    --x0-samples 100 \\
    --outdir pyident_results/x0_boxplot \\
    --seed 12345
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - import guard for optional dependency
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without matplotlib
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

import numpy as np
import pandas as pd

from ..metrics import pbh_margin_structured


def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample on the unit sphere S^{n-1}."""
    v = rng.standard_normal(n)
    nrm = float(np.linalg.norm(v))
    return v / (nrm if nrm > 0.0 else 1.0)


def sample_x0(
    n: int,
    rng: np.random.Generator,
    x0_density: float,
) -> np.ndarray:
    """Sample x0 with given sparsification level.
    
    Parameters
    ----------
    n : int
        Dimension of x0
    rng : np.random.Generator
        Random number generator
    x0_density : float
        Sparsification level. 1.0 = no sparsification (full sphere).
        < 1.0: apply Bernoulli mask with keep probability = x0_density, then renormalize.
    
    Returns
    -------
    np.ndarray
        Sampled x0 vector
    """
    x0 = sample_unit_sphere(n, rng)
    
    if x0_density < 1.0:
        # Apply Bernoulli sparsification: keep each entry with probability x0_density
        mask = rng.random(n) < x0_density
        x0 = x0 * mask
        
        # Renormalize to unit norm
        nrm = float(np.linalg.norm(x0))
        if nrm > 0.0:
            x0 = x0 / nrm
    
    return x0


def format_density_label(density: float) -> str:
    """Format density value for display."""
    if density >= 1.0:
        return "sphere"
    return f"ρ={density:g}"


def ensure_output_dir(path: pathlib.Path) -> pathlib.Path:
    """Create output directory and plots subdirectory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    outdir = ensure_output_dir(pathlib.Path(args.outdir))

    # Load dataset
    systems_df = pd.read_csv(args.dataset_csv)
    matrices = np.load(args.dataset_npz, allow_pickle=True)
    A_list = matrices["A"]
    B_list = matrices["B"]
    sys_idx = matrices["system_index"]
    index_map = {int(idx): pos for pos, idx in enumerate(sys_idx)}

    # Filter by state dimension and density range if provided
    if args.ndim_min is not None or args.ndim_max is not None:
        ndim_min = args.ndim_min if args.ndim_min is not None else 0
        ndim_max = args.ndim_max if args.ndim_max is not None else float("inf")
        systems_df = systems_df[(systems_df["n"] >= ndim_min) & (systems_df["n"] <= ndim_max)]

    if args.density_min is not None or args.density_max is not None:
        density_min = args.density_min if args.density_min is not None else 0.0
        density_max = args.density_max if args.density_max is not None else 1.0
        
        density_col = {
            "A": "meta_density_A",
            "B": "meta_density_B",
            "AB": "meta_density_AB",
        }.get(args.density_source, "meta_density_AB")
        
        systems_df = systems_df[
            (systems_df[density_col] >= density_min) & (systems_df[density_col] <= density_max)
        ]

    if systems_df.empty:
        raise RuntimeError("No systems matched the filtering criteria")

    # Sort x0 densities
    x0_densities = sorted([float(d) for d in args.x0_densities])

    # Generate records: (score, x0_density, n, m, etc.)
    records: list[dict[str, Any]] = []

    for _, row in systems_df.iterrows():
        sys_id = int(row["system_index"])
        pos = index_map.get(sys_id)
        if pos is None:
            raise RuntimeError(f"system_index {sys_id} missing from matrices file")
        
        A = A_list[pos]
        B = B_list[pos]
        n_cur = int(row["n"])
        m_cur = int(row["m"])

        # Sample x0 for each density level
        for x0_density in x0_densities:
            for _ in range(args.x0_samples):
                x0 = sample_x0(n_cur, rng, x0_density)
                
                # Compute PBH score
                pbh_score = float(pbh_margin_structured(A, B, x0))
                
                rec = {
                    "pbh": pbh_score,
                    "x0_density": float(x0_density),
                    "x0_density_label": format_density_label(x0_density),
                    "n": n_cur,
                    "m": m_cur,
                    "system_index": sys_id,
                }
                # Copy over metadata columns
                for col in row.index:
                    if col.startswith("meta_") or col in ["density"]:
                        rec[col] = row[col]
                records.append(rec)

    scores_df = pd.DataFrame(records)
    scores_path = outdir / "pbh_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    # Create boxplot
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting; please install it to run this experiment"
        ) from _MATPLOTLIB_IMPORT_ERROR

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Prepare data for boxplot
    data_by_density = []
    labels_by_density = []
    for x0_density in x0_densities:
        sub = scores_df[scores_df["x0_density"] == x0_density]
        pbh_vals = sub["pbh"].to_numpy()
        pbh_vals = pbh_vals[np.isfinite(pbh_vals)]  # Remove NaN/Inf
        data_by_density.append(pbh_vals)
        labels_by_density.append(format_density_label(x0_density))

    # Create boxplot
    bp = ax.boxplot(
        data_by_density,
        labels=labels_by_density,
        patch_artist=True,
        widths=0.6,
    )

    # Style boxes
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.set_xlabel("x0 Sampling Scheme", fontsize=14)
    ax.set_ylabel("PBH Score d(A,B|x0)", fontsize=14)
    ax.set_title("PBH Score Distribution vs x0 Sparsification Level", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = outdir / "plots" / "pbh_boxplot.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Scores saved to {scores_path}")
    print(f"✓ Plot saved to {plot_path}")
    print(f"✓ Total (A,B) systems: {len(systems_df)}")
    print(f"✓ x0 densities: {x0_densities}")
    print(f"✓ x0 samples per density: {args.x0_samples}")
    print(f"✓ Total (A,B,x0) triples: {len(scores_df)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-csv",
        required=True,
        help="Systems CSV (from sim_regcomb_ctrb or filter_unctrb_dataset)",
    )
    parser.add_argument(
        "--dataset-npz",
        required=True,
        help="Systems matrices NPZ (from sim_regcomb_ctrb or filter_unctrb_dataset)",
    )

    parser.add_argument(
        "--ndim-min",
        type=int,
        default=None,
        help="minimum state dimension to include",
    )
    parser.add_argument(
        "--ndim-max",
        type=int,
        default=None,
        help="maximum state dimension to include",
    )

    parser.add_argument(
        "--density-min",
        type=float,
        default=None,
        help="minimum density to include",
    )
    parser.add_argument(
        "--density-max",
        type=float,
        default=None,
        help="maximum density to include",
    )
    parser.add_argument(
        "--density-source",
        choices=["A", "B", "AB"],
        default="AB",
        help="which density metric to filter on",
    )

    parser.add_argument(
        "--x0-densities",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="list of x0 sparsification levels (1.0 = no sparsification)",
    )
    parser.add_argument(
        "--x0-samples",
        type=int,
        default=100,
        help="number of x0 samples per density level per (A,B) system",
    )

    parser.add_argument("--outdir", default="results/x0_boxplot", help="output directory")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
