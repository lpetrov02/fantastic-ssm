# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm import Mamba, ops
from mamba_ssm.ops.triton.layer_norm import RMSNorm

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn
except ImportError:
    layer_norm_fn, rms_norm_fn = None, None


class MambaBlock(nn.Module):
    """Один блок: RMSNorm → Mamba → residual"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class Mamba130M(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pad_vocab_size_multiple: int = 8,
    ):
        super().__init__()

        # Выравниваем vocab_size до кратного 8 (для эффективности)
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        self.norm_f = RMSNorm(d_model)  # финальная нормализация
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying — стандартная практика
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        # Mamba инициализирует себя сама внутри

    def forward(self, input_ids):
        x = self.embedding(input_ids)      # (B, L, d_model)

        for layer in self.layers:
            x = layer(x)                   # (B, L, d_model)

        x = self.norm_f(x)
        logits = self.lm_head(x)           # (B, L, vocab_size)
        return logits
