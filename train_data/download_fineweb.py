"""
FineWeb-Edu subset downloader
Скачивает ровно N токенов через streaming, без загрузки всего датасета.

Использование:
    python download_fineweb.py --tokens 5e9 --output_dir ./data/fineweb --split train
    python download_fineweb.py --tokens 100e6 --output_dir ./data/fineweb_val --split train --offset 9_000_000_000

Зависимости:
    pip install datasets tiktoken numpy tqdm
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tiktoken
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Stream and tokenize FineWeb-Edu subset")
    parser.add_argument(
        "--tokens", type=float, default=5e9,
        help="Сколько токенов скачать (например: 5e9 = 5B)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/fineweb",
        help="Директория для сохранения .npy шардов"
    )
    parser.add_argument(
        "--shard_size", type=float, default=1e8,
        help="Токенов на один шард (default: 100M)"
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Пропустить первые N токенов (для val сплита из того же датасета)"
    )
    parser.add_argument(
        "--name", type=str, default="sample-10BT",
        choices=["sample-10BT", "sample-100BT", "sample-350BT", "default"],
        help="Какой сабсет FineWeb-Edu использовать"
    )
    parser.add_argument(
        "--num_proc", type=int, default=4,
        help="Параллельных процессов токенизации"
    )
    parser.add_argument(
        "--dtype", type=str, default="uint16",
        choices=["uint16", "uint32"],
        help="dtype для токенов (uint16 хватает для GPT-2 vocab=50257)"
    )
    return parser.parse_args()


def get_tokenizer():
    """GPT-2 tokenizer через tiktoken"""
    enc = tiktoken.get_encoding("gpt2")
    return enc


def tokenize_document(text: str, enc) -> np.ndarray:
    """Токенизирует один документ, добавляет EOT в конце"""
    EOT = enc._special_tokens["<|endoftext|>"]
    tokens = enc.encode_ordinary(text)
    tokens.append(EOT)
    return np.array(tokens, dtype=np.uint16)


def save_shard(tokens: list, shard_idx: int, output_dir: Path, dtype: str):
    """Сохраняет шард как .npy файл"""
    arr = np.concatenate(tokens).astype(dtype)
    path = output_dir / f"shard_{shard_idx:05d}.npy"
    np.save(path, arr)
    return len(arr), path


def main():
    args = parse_args()

    target_tokens = int(args.tokens)
    shard_size = int(args.shard_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== FineWeb-Edu Downloader ===")
    print(f"  Датасет:       HuggingFaceFW/fineweb-edu [{args.name}]")
    print(f"  Цель:          {target_tokens:,} токенов ({target_tokens/1e9:.1f}B)")
    print(f"  Размер шарда:  {shard_size:,} токенов ({shard_size/1e6:.0f}M)")
    print(f"  Offset:        {args.offset:,} токенов")
    print(f"  Output:        {output_dir}")
    print(f"  dtype:         {args.dtype}")
    print()

    # Загружаем датасет в streaming режиме — не скачивает всё
    print("Инициализируем streaming dataset...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.name,
        split="train",
        streaming=True,
    )

    enc = get_tokenizer()

    total_tokens = 0
    shard_tokens = []
    shard_token_count = 0
    shard_idx = 0
    doc_count = 0
    skipped_tokens = 0
    start_time = time.time()

    # Метаданные для записи
    meta = {
        "total_tokens": 0,
        "shards": [],
        "dataset": f"HuggingFaceFW/fineweb-edu/{args.name}",
        "tokenizer": "gpt2",
        "offset_tokens": args.offset,
    }

    pbar = tqdm(
        total=target_tokens,
        unit="tok",
        unit_scale=True,
        desc="Токенизация",
        dynamic_ncols=True,
    )

    for doc in ds:
        text = doc.get("text", "")
        if not text:
            continue

        tokens = tokenize_document(text, enc)
        doc_count += 1

        # Если нужен offset — пропускаем токены в начале
        if skipped_tokens < args.offset:
            remaining_to_skip = args.offset - skipped_tokens
            if len(tokens) <= remaining_to_skip:
                skipped_tokens += len(tokens)
                continue
            else:
                tokens = tokens[remaining_to_skip:]
                skipped_tokens = args.offset

        # Добавляем токены в текущий шард
        # Если документ переполняет шард — разрезаем
        while len(tokens) > 0:
            space_in_shard = shard_size - shard_token_count
            chunk = tokens[:space_in_shard]
            tokens = tokens[space_in_shard:]

            shard_tokens.append(chunk)
            shard_token_count += len(chunk)
            total_tokens += len(chunk)
            pbar.update(len(chunk))

            # Шард заполнен — сохраняем
            if shard_token_count >= shard_size:
                n_saved, path = save_shard(shard_tokens, shard_idx, output_dir, args.dtype)
                meta["shards"].append({"idx": shard_idx, "path": str(path), "tokens": n_saved})
                elapsed = time.time() - start_time
                speed = total_tokens / elapsed / 1e6
                tqdm.write(
                    f"  Шард {shard_idx:03d} сохранён: {n_saved/1e6:.1f}M токенов → {path.name}"
                    f"  [{speed:.1f}M tok/s, {elapsed/60:.1f} мин]"
                )
                shard_tokens = []
                shard_token_count = 0
                shard_idx += 1

        if total_tokens >= target_tokens:
            break

    pbar.close()

    # Сохраняем последний неполный шард
    if shard_tokens:
        n_saved, path = save_shard(shard_tokens, shard_idx, output_dir, args.dtype)
        meta["shards"].append({"idx": shard_idx, "path": str(path), "tokens": n_saved})
        print(f"  Последний шард {shard_idx:03d}: {n_saved/1e6:.1f}M токенов → {path.name}")

    # Сохраняем метаданные
    meta["total_tokens"] = total_tokens
    import json
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start_time
    print()
    print(f"=== Готово ===")
    print(f"  Всего токенов:  {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"  Документов:     {doc_count:,}")
    print(f"  Шардов:         {shard_idx + 1}")
    print(f"  Время:          {elapsed/60:.1f} мин")
    print(f"  Скорость:       {total_tokens/elapsed/1e6:.1f}M tok/s")
    print(f"  Метаданные:     {meta_path}")
    print()
    print(f"  Размер на диске: ~{total_tokens * (2 if args.dtype == 'uint16' else 4) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
