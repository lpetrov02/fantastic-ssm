#!/usr/bin/env bash
# Example NIAH evaluation runs.  Run from the project root.

set -euo pipefail

# CHECKPOINT="/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/mamba-niah-ft/step_0001000.pt"
# CHECKPOINT="/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/fantastic-130m-e16a16-orth001-pretrained/step_0027000.pt"
# CHECKPOINT="/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/attentive-130m-pretrained/step_0006000.pt"
CHECKPOINT="/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/attentive-130m/step_0040000.pt"
OUT_DIR="evaluation/results/niah"
DEVICE="cuda"

# ── single model ──────────────────────────────────────────────────────────────
python niah_eval.py \
    --checkpoint "$CHECKPOINT" \
    --output     "$OUT_DIR/pupupu.json" \
    --context_lengths 512 1024 2048 \
    --n_depths   9 \
    --n_samples  100 \
    --batch_size 4 \
    --device     "$DEVICE" \
\
    # --output     "$OUT_DIR/niah_mamba_130m.json" \

# ── plot ──────────────────────────────────────────────────────────────────────
# python evaluation/plot_results.py \
#     "$OUT_DIR/niah_$(basename "$CHECKPOINT" .pt).json" \
#     --metric loss

# python evaluation/plot_results.py \
#     "$OUT_DIR/niah_$(basename "$CHECKPOINT" .pt).json" \
#     --metric ppl

echo "Done. Results in $OUT_DIR/"
