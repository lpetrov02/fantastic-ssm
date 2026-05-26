"""
Plot NIAH results as a heatmap.

Usage:
    python evaluation/plot_results.py results/niah_fantastic.json
    python evaluation/plot_results.py results/niah_fantastic.json --metric ppl --out niah.png
    python evaluation/plot_results.py a.json b.json --labels Fantastic Mamba
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_results(path: str, metric: str):
    with open(path) as f:
        data = json.load(f)

    raw = data["results"]
    context_lengths = sorted(int(k) for k in raw)
    depth_pcts = sorted(float(k) for k in next(iter(raw.values())))

    # matrix[i][j] = metric at (context_lengths[i], depth_pcts[j])
    key = "mean_ppl" if metric == "ppl" else "mean_loss"
    matrix = np.array([
        [raw[str(ctx)][str(d)][key] for d in depth_pcts]
        for ctx in context_lengths
    ])

    meta = {
        "model": data.get("model", Path(path).stem),
        "step": data.get("model_step"),
        "checkpoint": data.get("checkpoint", ""),
    }
    return matrix, context_lengths, depth_pcts, meta


def plot_single(ax, matrix, context_lengths, depth_pcts, title, metric, vmin, vmax):
    label = "Perplexity" if metric == "ppl" else "Loss"
    cmap = "RdYlGn_r"  # red = high (bad), green = low (good)

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )

    # Axes
    ax.set_xticks(range(len(depth_pcts)))
    ax.set_xticklabels([f"{d:.0%}" for d in depth_pcts], fontsize=8)
    ax.set_yticks(range(len(context_lengths)))
    ax.set_yticklabels(context_lengths, fontsize=8)
    ax.set_xlabel("Needle depth (position in context)", fontsize=9)
    ax.set_ylabel("Context length (tokens)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)

    # Cell annotations
    for i in range(len(context_lengths)):
        for j in range(len(depth_pcts)):
            val = matrix[i, j]
            text = f"{val:.1f}" if metric == "ppl" else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="black" if 0.3 < (val - vmin) / max(vmax - vmin, 1e-6) < 0.7 else "white")

    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+", help="Path(s) to niah JSON result files")
    p.add_argument("--metric", choices=["loss", "ppl"], default="loss",
                   help="Which metric to plot")
    p.add_argument("--labels", nargs="*", help="Override legend labels (one per file)")
    p.add_argument("--out", default=None,
                   help="Output image path (default: same dir as first result file)")
    args = p.parse_args()

    all_data = []
    for path in args.results:
        matrix, ctx_lens, depths, meta = load_results(path, args.metric)
        all_data.append((matrix, ctx_lens, depths, meta))

    # Shared color scale across all subplots
    vmin = min(d[0].min() for d in all_data)
    vmax = max(d[0].max() for d in all_data)

    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n + 1, 4), squeeze=False)
    axes = axes[0]

    labels = args.labels or [None] * n
    for ax, (matrix, ctx_lens, depths, meta), label in zip(axes, all_data, labels):
        if label is None:
            label = meta["model"]
            if meta.get("step"):
                label += f" (step {meta['step']})"
        im = plot_single(ax, matrix, ctx_lens, depths, label, args.metric, vmin, vmax)

    # Shared colorbar
    cbar_label = "Perplexity" if args.metric == "ppl" else "Cross-entropy loss"
    fig.colorbar(im, ax=axes[-1], label=cbar_label, fraction=0.046, pad=0.04)

    metric_str = args.metric
    fig.suptitle(
        f"Needle-in-a-Haystack — passkey retrieval ({metric_str})",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        stem = "_vs_".join(Path(r).stem for r in args.results)
        out_path = Path(args.results[0]).parent / f"{stem}_{metric_str}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
