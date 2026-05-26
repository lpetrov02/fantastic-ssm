#!/usr/bin/env bash
# Example NIAH evaluation runs.  Run from the project root.

set -euo pipefail

CHECKPOINT="${1:-checkpoints/step_10000.pt}"
OUT_DIR="evaluation/results"
DEVICE="${DEVICE:-cuda}"

# ── single model ──────────────────────────────────────────────────────────────
python evaluation/niah_eval.py \
    --checkpoint "$CHECKPOINT" \
    --output     "$OUT_DIR/niah_$(basename "$CHECKPOINT" .pt).json" \
    --context_lengths 512 1024 2048 \
    --n_depths   9 \
    --n_samples  10 \
    --batch_size 4 \
    --device     "$DEVICE"

# ── plot ──────────────────────────────────────────────────────────────────────
python evaluation/plot_results.py \
    "$OUT_DIR/niah_$(basename "$CHECKPOINT" .pt).json" \
    --metric loss

python evaluation/plot_results.py \
    "$OUT_DIR/niah_$(basename "$CHECKPOINT" .pt).json" \
    --metric ppl

echo "Done. Results in $OUT_DIR/"
