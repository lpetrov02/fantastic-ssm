"""
Run NIAH evaluation on multiple checkpoints and produce a comparison plot.

Usage:
    python evaluation/compare_models.py \\
        --checkpoints checkpoints/fantastic_10k.pt checkpoints/mamba_10k.pt \\
        --labels Fantastic Mamba \\
        --output evaluation/results/comparison \\
        [--context_lengths 512 1024 2048] \\
        [--n_depths 9] \\
        [--n_samples 10] \\
        [--batch_size 4]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Compare multiple models on NIAH")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoint .pt files (one per model)")
    p.add_argument("--labels", nargs="*",
                   help="Display labels (one per checkpoint)")
    p.add_argument("--output", default="evaluation/results/comparison",
                   help="Directory for JSON results and plots")
    p.add_argument("--context_lengths", nargs="+", type=int,
                   default=[512, 1024, 2048])
    p.add_argument("--n_depths", type=int, default=9)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--metric", choices=["loss", "ppl"], default="loss")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_files = []
    for ckpt in args.checkpoints:
        stem = Path(ckpt).stem
        json_path = out_dir / f"niah_{stem}.json"
        result_files.append(str(json_path))

        if json_path.exists():
            print(f"Skipping {ckpt} (result already exists at {json_path})")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, "evaluation/niah_eval.py",
            "--checkpoint", ckpt,
            "--output", str(json_path),
            "--context_lengths", *map(str, args.context_lengths),
            "--n_depths", str(args.n_depths),
            "--n_samples", str(args.n_samples),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
        ]
        subprocess.run(cmd, check=True)

    print(f"\n{'='*60}")
    print("Plotting …")
    plot_cmd = [
        sys.executable, "evaluation/plot_results.py",
        *result_files,
        "--metric", args.metric,
        "--out", str(out_dir / f"comparison_{args.metric}.png"),
    ]
    if args.labels:
        plot_cmd += ["--labels", *args.labels]
    subprocess.run(plot_cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
