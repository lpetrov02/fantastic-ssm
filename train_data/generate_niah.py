"""
Pre-generate a static NIAH fine-tuning dataset and save to disk.

Generates passkey-retrieval samples using a seed space completely disjoint
from the evaluation grid (eval uses base_seed=42; we use 1337).

Output layout
-------------
  <output_dir>/
    shard_00000.npz   — up to --shard_size samples each
    shard_00001.npz
    ...
    meta.json

Each .npz contains two arrays:
    inputs  : int64  [N, max_len]  — token ids, padded with 0
    labels  : int64  [N, max_len]  — answer token ids at answer positions, -100 elsewhere

Usage
-----
    python train_data/generate_niah.py \\
        --output_dir train_data/niah_train \\
        --context_lengths 512 1024 2048 \\
        --n_depths 9 \\
        --n_samples 200 \\
        --shard_size 1000

Then load with NIAHShardedDataset (defined below) or read the .npz files directly.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.niah_data import build_sample
from train_data.niah_dataset import TRAIN_BASE_SEED, _LCG_MUL


# ── generation ────────────────────────────────────────────────────────────────

def _build_index(context_lengths: List[int], depth_pcts: List[float], n_samples: int):
    return [
        (ctx_len, depth, i)
        for ctx_len in context_lengths
        for depth in depth_pcts
        for i in range(n_samples)
    ]


def generate_dataset(
    tokenizer,
    context_lengths: List[int],
    n_depths: int,
    n_samples: int,
    base_seed: int = TRAIN_BASE_SEED,
) -> List[dict]:
    """
    Returns a list of dicts with keys: inputs (list[int]), labels (list[int]).
    Labels are -100 except at answer positions.
    """
    depth_pcts = [round((i + 1) / n_depths, 4) for i in range(n_depths)]
    index = _build_index(context_lengths, depth_pcts, n_samples)

    records = []
    for idx, (ctx_len, depth, _) in enumerate(tqdm(index, desc="Generating")):
        seed = (base_seed + idx * _LCG_MUL) & 0x7FFFFFFF
        sample = build_sample(tokenizer, context_length=ctx_len, depth_pct=depth, seed=seed)

        labels = [-100] * len(sample.input_ids)
        for pos in range(sample.answer_start, sample.answer_end):
            labels[pos] = sample.input_ids[pos]

        records.append({"inputs": sample.input_ids, "labels": labels})

    return records


# ── saving ────────────────────────────────────────────────────────────────────

def save_shards(records: List[dict], output_dir: Path, shard_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    shards_meta = []
    n_shards = (len(records) + shard_size - 1) // shard_size

    for shard_idx in range(n_shards):
        chunk = records[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        max_len = max(len(r["inputs"]) for r in chunk)

        inputs_arr = np.zeros((len(chunk), max_len), dtype=np.int64)
        labels_arr = np.full((len(chunk), max_len), -100, dtype=np.int64)

        for i, r in enumerate(chunk):
            L = len(r["inputs"])
            inputs_arr[i, :L] = r["inputs"]
            labels_arr[i, :L] = r["labels"]

        fname = f"shard_{shard_idx:05d}.npz"
        np.savez_compressed(output_dir / fname, inputs=inputs_arr, labels=labels_arr)
        shards_meta.append({"idx": shard_idx, "path": fname, "n_samples": len(chunk)})
        print(f"  Saved {fname}  ({len(chunk)} samples, max_len={max_len})")

    return shards_meta


# ── dataloader for pre-generated shards ──────────────────────────────────────

class NIAHShardedDataset:
    """
    Iterable dataset over pre-generated .npz shards.

    Yields (input_ids, labels) tensors, each shape [seq_len].
    Use with torch.utils.data.DataLoader + collate_niah for batching.

    Example
    -------
        from torch.utils.data import DataLoader
        from train_data.niah_dataset import collate_niah
        from train_data.generate_niah import NIAHShardedDataset

        ds = NIAHShardedDataset("train_data/niah_train")
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_niah)
    """

    import torch  # local import so the module is importable without torch in gen script

    def __init__(self, data_dir: str):
        import torch
        self._torch = torch
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.shard_paths = [self.data_dir / s["path"] for s in self.meta["shards"]]

    def __len__(self) -> int:
        return sum(s["n_samples"] for s in self.meta["shards"])

    def __iter__(self):
        torch = self._torch
        for shard_path in self.shard_paths:
            data = np.load(shard_path)
            inputs = data["inputs"]   # [N, max_len]
            labels = data["labels"]   # [N, max_len]
            for i in range(len(inputs)):
                # Strip trailing padding (0 in inputs) to keep sequences compact
                L = int((inputs[i] != 0).sum()) or 1
                yield (
                    torch.from_numpy(inputs[i, :L].copy()),
                    torch.from_numpy(labels[i, :L].copy()),
                )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate static NIAH fine-tuning dataset")
    p.add_argument("--output_dir", default="train_data/niah_train",
                   help="Directory to write shards and meta.json")
    p.add_argument("--context_lengths", nargs="+", type=int,
                   default=[512, 1024, 2048],
                   help="Context lengths to include (tokens)")
    p.add_argument("--n_depths", type=int, default=9,
                   help="Number of evenly-spaced needle depths in (0, 1]")
    p.add_argument("--n_samples", type=int, default=200,
                   help="Repeats per (context_length, depth) cell")
    p.add_argument("--shard_size", type=int, default=1000,
                   help="Samples per output shard")
    p.add_argument("--base_seed", type=int, default=TRAIN_BASE_SEED,
                   help="Root seed (must NOT be 42 — that is the eval seed)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.base_seed == 42:
        raise ValueError(
            "base_seed=42 is reserved for evaluation. "
            "Use a different seed to avoid data leakage."
        )

    output_dir = Path(args.output_dir)
    depth_pcts = [round((i + 1) / args.n_depths, 4) for i in range(args.n_depths)]

    print(f"Output dir     : {output_dir}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Depth positions: {depth_pcts}")
    print(f"Samples/cell   : {args.n_samples}")
    print(f"Train base seed: {args.base_seed}  (eval seed=42, kept separate)")
    total = len(args.context_lengths) * len(depth_pcts) * args.n_samples
    print(f"Total samples  : {total}")
    print()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    records = generate_dataset(
        tokenizer=tokenizer,
        context_lengths=args.context_lengths,
        n_depths=args.n_depths,
        n_samples=args.n_samples,
        base_seed=args.base_seed,
    )

    shards_meta = save_shards(records, output_dir, args.shard_size)

    meta = {
        "n_samples": len(records),
        "context_lengths": args.context_lengths,
        "depth_pcts": depth_pcts,
        "n_samples_per_cell": args.n_samples,
        "base_seed": args.base_seed,
        "eval_seed": 42,
        "tokenizer": "gpt2",
        "shards": shards_meta,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {len(records)} samples in {len(shards_meta)} shards → {output_dir}")
    print(f"Load with NIAHShardedDataset('{output_dir}')")


if __name__ == "__main__":
    main()
