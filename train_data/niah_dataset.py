"""
NIAH fine-tuning dataset — on-the-fly generation, zero eval leakage.

Eval uses base_seed=42 and hash((ctx_len, depth, i)) for seeds.
Training uses base_seed=1337 and a linear-index seed space — completely disjoint.

Each sample:
    input_ids : [context_tokens ... answer_tokens]
    labels    : -100 at every context position, answer token ids at answer positions
                (cross_entropy ignores -100 → loss computed only on answer tokens)

Quick start:
    from train_data.niah_dataset import make_niah_dataloader
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    loader = make_niah_dataloader(tok, context_lengths=[512, 1024, 2048])
    for input_ids, labels in loader:
        loss = model(input_ids, targets=labels)  # model must handle -100 targets
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.niah_data import build_sample

# Eval default seed is 42. Training uses 1337 to stay in a disjoint seed space.
TRAIN_BASE_SEED = 1337

# LCG multiplier — maps linear sample index to a well-spread seed value.
_LCG_MUL = 6364136223846793005


class NIAHDataset(Dataset):
    """
    Generates passkey-retrieval (NIAH) samples for fine-tuning.

    Context lengths and depth percentages are enumerated exhaustively;
    n_samples repeats each (ctx_len, depth) cell with a distinct seed.

    Parameters
    ----------
    tokenizer      : HuggingFace tokenizer (GPT-2 recommended)
    n_samples      : repeats per (context_length, depth_pct) cell
    context_lengths: list of total context sizes in tokens (haystack + needle + query)
    n_depths       : number of evenly-spaced needle-depth positions in (0, 1]
    base_seed      : root seed — do NOT use 42 (that is the eval seed)
    """

    def __init__(
        self,
        tokenizer,
        n_samples: int = 50,
        context_lengths: Optional[List[int]] = None,
        n_depths: int = 9,
        base_seed: int = TRAIN_BASE_SEED,
    ):
        if context_lengths is None:
            context_lengths = [512, 1024, 2048]

        self.tokenizer = tokenizer
        self.base_seed = base_seed

        depth_pcts = [round((i + 1) / n_depths, 4) for i in range(n_depths)]

        # Flat index: each entry is (ctx_len, depth_pct, repeat_i)
        self.index: List[tuple] = [
            (ctx_len, depth, i)
            for ctx_len in context_lengths
            for depth in depth_pcts
            for i in range(n_samples)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ctx_len, depth, _ = self.index[idx]

        # Seed: linear index spread via LCG — guaranteed distinct per sample
        # and independent from eval's hash((ctx_len, depth, i)) formula.
        seed = (self.base_seed + idx * _LCG_MUL) & 0x7FFFFFFF

        sample = build_sample(
            self.tokenizer,
            context_length=ctx_len,
            depth_pct=depth,
            seed=seed,
        )

        input_ids = torch.tensor(sample.input_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        labels[sample.answer_start : sample.answer_end] = (
            input_ids[sample.answer_start : sample.answer_end]
        )

        return input_ids, labels


def collate_niah(batch):
    """
    Pad a batch of variable-length NIAH samples to the maximum length.
    input_ids is padded with 0; labels with -100 (ignored by cross_entropy).
    """
    input_ids_list, labels_list = zip(*batch)
    max_len = max(x.size(0) for x in input_ids_list)

    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)

    for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
        L = ids.size(0)
        input_ids[i, :L] = ids
        labels[i, :L] = lbl

    return input_ids, labels


def make_niah_dataloader(
    tokenizer,
    n_samples: int = 50,
    context_lengths: Optional[List[int]] = None,
    n_depths: int = 9,
    batch_size: int = 4,
    base_seed: int = TRAIN_BASE_SEED,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience wrapper: creates NIAHDataset and wraps it in a DataLoader.

    Returns batches of (input_ids, labels), each shape [batch_size, max_len].
    Pass labels directly to cross_entropy — -100 positions are ignored automatically.
    """
    if context_lengths is None:
        context_lengths = [512, 1024, 2048]

    dataset = NIAHDataset(
        tokenizer=tokenizer,
        n_samples=n_samples,
        context_lengths=context_lengths,
        n_depths=n_depths,
        base_seed=base_seed,
    )
    print(
        f"NIAHDataset: {len(dataset)} samples, "
        f"ctx_lengths={context_lengths}, n_depths={n_depths}"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_niah,
        num_workers=num_workers,
    )
