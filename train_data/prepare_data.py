"""
Подготовка train/val сплитов для FineWeb-Edu.
Запускает download_fineweb.py дважды: train из первых 5B, val из следующих 100M.

Использование:
    bash prepare_data.sh
"""

# prepare_data.sh  (сохрани как .sh и запусти)
SCRIPT = """#!/bin/bash
set -e

TRAIN_TOKENS=5_000_000_000   # 5B токенов для train
VAL_TOKENS=100_000_000        # 100M токенов для val (берём после train offset)
VAL_OFFSET=5_000_000_000      # val начинается после train

echo "=== Скачиваем TRAIN (5B токенов) ==="
python download_fineweb.py \\
    --tokens $TRAIN_TOKENS \\
    --output_dir ./data/train \\
    --shard_size 100_000_000 \\
    --offset 0 \\
    --name sample-10BT

echo ""
echo "=== Скачиваем VAL (100M токенов, offset=5B) ==="
python download_fineweb.py \\
    --tokens $VAL_TOKENS \\
    --output_dir ./data/val \\
    --shard_size 100_000_000 \\
    --offset $VAL_OFFSET \\
    --name sample-10BT

echo ""
echo "=== Итог ==="
echo "Train шарды: $(ls ./data/train/shard_*.npy | wc -l)"
echo "Val шарды:   $(ls ./data/val/shard_*.npy | wc -l)"
du -sh ./data/train ./data/val
"""

with open("scripts/prepare_data.sh", "w") as f:
    f.write(SCRIPT)

print("Сохранено: prepare_data.sh")
print("Запуск: bash prepare_data.sh")
