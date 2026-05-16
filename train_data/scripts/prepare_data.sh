#!/bin/bash
set -e

TRAIN_TOKENS=5_000_000_000   # 5B токенов для train
VAL_TOKENS=100_000_000        # 100M токенов для val (берём после train offset)
VAL_OFFSET=5_000_000_000      # val начинается после train

echo "=== Скачиваем TRAIN (5B токенов) ==="
python download_fineweb.py \
    --tokens $TRAIN_TOKENS \
    --output_dir /home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/train_data/data/train \
    --shard_size 100_000_000 \
    --offset 0 \
    --name sample-10BT

echo ""
echo "=== Скачиваем VAL (100M токенов, offset=5B) ==="
python download_fineweb.py \
    --tokens $VAL_TOKENS \
    --output_dir /home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/train_data/data/val \
    --shard_size 100_000_000 \
    --offset $VAL_OFFSET \
    --name sample-10BT

echo ""
echo "=== Итог ==="
echo "Train шарды: $(ls ./data/train/shard_*.npy | wc -l)"
echo "Val шарды:   $(ls ./data/val/shard_*.npy | wc -l)"
du -sh ./data/train ./data/val
