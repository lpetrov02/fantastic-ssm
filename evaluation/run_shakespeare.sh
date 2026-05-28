#!/usr/bin/env bash
# Example NIAH evaluation runs.  Run from the project root.

set -euo pipefail

# CHECKPOINT="/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/mamba-130m/step_0020000.pt"
# CHECKPOINT="/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/fantastic-130m-e16a16-orth001-pretrained/step_0023000.pt"
CHECKPOINT="/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/attentive-130m-pretrained/step_0006000.pt"
OUT_DIR="evaluation/results/shakespeare"
DEVICE="cuda"

# ── single model ──────────────────────────────────────────────────────────────
python shakespeare_eval.py \
    --checkpoint "$CHECKPOINT" \
    --output     "$OUT_DIR/shakespeare_fantastic.json"

echo "Done. Results in $OUT_DIR/"
