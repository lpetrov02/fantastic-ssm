"""
DataLoader для обучения из предтокенизированных .npy шардов.
Совместим с DDP (каждый ранк читает свою часть данных).

Использование:
    from data_loader import ShardedDataLoader

    loader = ShardedDataLoader(
        data_dir="./data/train",
        seq_len=2048,
        batch_size=16,       # на один GPU
        rank=0,
        world_size=8,
    )

    for x, y in loader:
        # x, y: [batch_size, seq_len] int64 tensors
        loss = model(x, targets=y)
"""

import os
import json
import math
import numpy as np
import torch
from pathlib import Path
from typing import Iterator, Tuple


class ShardedDataLoader:
    """
    Загружает .npy шарды по одному, нарезает на батчи seq_len.
    DDP-совместим: каждый ранк берёт свои батчи без пересечений.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 2048,
        batch_size: int = 16,
        rank: int = 0,
        world_size: int = 1,
        shuffle_shards: bool = True,
        seed: int = 42,
        offset: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.offset = offset

        # Ищем все шарды
        self.shards = sorted(self.data_dir.glob("shard_*.npy"))
        assert len(self.shards) > 0, f"Нет шардов в {data_dir}"

        # Читаем мету если есть
        meta_path = self.data_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self._epoch = 0
        print(
            f"[Rank {rank}] DataLoader: {len(self.shards)} шардов в {data_dir}, "
            f"seq_len={seq_len}, batch_size={batch_size}"
        )

    def _get_shard_order(self) -> list:
        """Порядок шардов для текущей эпохи (shuffle по эпохе)"""
        indices = list(range(len(self.shards)))
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(indices)
        return indices

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        shard_order = self._get_shard_order()

        for shard_idx in shard_order:
            shard_path = self.shards[shard_idx]
            tokens = np.load(shard_path).astype(np.int64)

            # Сколько полных seq_len окон влезает в шард
            n_tokens = len(tokens)
            n_seqs = (n_tokens - 1) // self.seq_len  # -1 т.к. нужен y = x[1:]

            if n_seqs == 0:
                continue

            # Каждый ранк берёт свои батчи через stride
            # Ранк 0: батчи 0, world_size, 2*world_size, ...
            # Ранк 1: батчи 1, world_size+1, ...
            global_batch_size = self.batch_size * self.world_size
            n_global_batches = n_seqs // global_batch_size

            if n_global_batches == 0:
                continue

            for batch_i in range(n_global_batches):
                while self.offset > 0:
                    self.offset -= 1
                    continue
                # Глобальные индексы всех sequences в этом батче
                global_seq_start = batch_i * global_batch_size

                # Индексы для этого ранка
                local_indices = [
                    global_seq_start + self.rank * self.batch_size + j
                    for j in range(self.batch_size)
                ]

                # Нарезаем токены
                xs = []
                ys = []
                for seq_idx in local_indices:
                    start = seq_idx * self.seq_len
                    end = start + self.seq_len + 1
                    if end > n_tokens:
                        break
                    chunk = tokens[start:end]
                    xs.append(chunk[:-1])
                    ys.append(chunk[1:])

                if len(xs) < self.batch_size:
                    break

                x = torch.from_numpy(np.stack(xs))  # [B, seq_len]
                y = torch.from_numpy(np.stack(ys))  # [B, seq_len]
                yield x, y

        self._epoch += 1

    def estimate_steps_per_epoch(self) -> int:
        """Примерное количество шагов на эпоху"""
        total_tokens = self.meta.get("total_tokens", 0)
        if total_tokens == 0:
            # Считаем из файлов
            total_tokens = sum(
                np.load(s, mmap_mode='r').shape[0] for s in self.shards
            )
        tokens_per_step = self.batch_size * self.world_size * self.seq_len
        return total_tokens // tokens_per_step


class ValDataLoader:
    """
    Простой загрузчик для валидации — читает весь val сет один раз.
    Не DDP-aware (каждый ранк считает одинаково, логируем с ранка 0).
    """

    def __init__(self, data_dir: str, seq_len: int = 2048, batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shards = sorted(self.data_dir.glob("shard_*.npy"))
        assert len(self.shards) > 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for shard_path in self.shards:
            tokens = np.load(shard_path).astype(np.int64)
            n_seqs = (len(tokens) - 1) // self.seq_len
            n_batches = n_seqs // self.batch_size

            for batch_i in range(n_batches):
                xs, ys = [], []
                for j in range(self.batch_size):
                    seq_idx = batch_i * self.batch_size + j
                    start = seq_idx * self.seq_len
                    end = start + self.seq_len + 1
                    if end > len(tokens):
                        break
                    xs.append(tokens[start:end - 1])
                    ys.append(tokens[start + 1:end])
                if len(xs) == self.batch_size:
                    yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


if __name__ == "__main__":
    # Быстрая проверка
    import tempfile

    print("Тест DataLoader...")

    # Создаём фейковый шард
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fake_tokens = np.random.randint(0, 50257, size=(10_000_000,), dtype=np.uint16)
        np.save(tmpdir / "shard_00000.npy", fake_tokens)

        loader = ShardedDataLoader(
            data_dir=str(tmpdir),
            seq_len=2048,
            batch_size=4,
            rank=0,
            world_size=1,
        )

        for i, (x, y) in enumerate(loader):
            print(f"  Батч {i}: x={x.shape}, y={y.shape}, dtype={x.dtype}")
            assert x.shape == (4, 2048)
            assert y.shape == (4, 2048)
            assert (x[:, 1:] == y[:, :-1]).all(), "y должен быть x сдвинутый на 1"
            if i >= 2:
                break

    print("OK!")
