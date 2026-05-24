# Copyright (c) 2023, Albert Gu, Tri Dao.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from einops import rearrange, repeat
from pydantic import validate_call


from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm import Mamba

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from utils.token_router.router import TokenTopKRouter


class Fantastic(nn.Module):

    def __init__(
        self,
        d_model,
        d_state: int = 16,
        d_conv:int = 4,
        expand: int = 2,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx = None,
        lb_strategy: str = "none",
        lb_coef: float = 0.01,
        aux_free_bias_step: float = 1e-3,
        dt_strategy: str = "random",
        device = None,
        dtype = None,
        **kwargs
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.dt_strategy = dt_strategy

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.d_conv = d_conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        ) if d_conv > 0 else nn.Identity()
        self.router = TokenTopKRouter(
            self.d_inner,
            self.num_experts,
            self.top_k,
            lb_strategy,
            lb_coef,
            aux_free_bias_step
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # DELTA
        if dt_strategy == "random":
            log_dt = torch.rand(self.num_experts, self.d_inner) \
                * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        elif dt_strategy == "linspace":
            dt_min = torch.linspace(dt_min, dt_max, self.num_experts + 1)[:-1, None]
            dt_max = torch.linspace(dt_min, dt_max, self.num_experts + 1)[1:, None]
            log_dt = torch.rand(self.num_experts, self.d_inner) \
                * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        elif dt_strategy == "logspace":
            dt_min = torch.logspace(dt_min, dt_max, self.num_experts + 1)[:-1, None]
            dt_max = torch.logspace(dt_min, dt_max, self.num_experts + 1)[1:, None]
            log_dt = torch.rand(self.num_experts, self.d_inner) \
                * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        else:
            raise ValueError(f"Unknown dt strategy: {dt_strategy}")
        self.log_dt = nn.Parameter(log_dt)

        # A-matrix
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # B, C, D
        B = torch.ones(self.num_experts, self.d_state, dtype=torch.float32)
        self.B = nn.Parameter(B)
        C = torch.randn(self.num_experts, self.d_state, dtype=torch.float32)
        self.C = nn.Parameter(C)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

    def forward(self,
        hidden_states,
        noise_scale: float = 1.0,
        mixture_temperature: float = 1.0,
        straight_through: bool = True,
        uniform_topk_eps: float = 0.0,
        inference_params=None
    ):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print(f"mixer: {type(hidden_states)}")
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)  # (B, L, H) each
        x, z = x.transpose(-1, -2), z.transpose(-1, -2)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Compute short convolution

        if conv_state is not None:
            conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
        if self.d_conv > 0:
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                weight = rearrange(self.conv1d.weight, "d 1 w -> d w")
                # print(x.shape, weight.shape)
                x = causal_conv1d_fn(
                    x,
                    weight=weight,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )

        alpha, _ = self.router(
            x,
            noise_scale=noise_scale,
            mixture_temperature=mixture_temperature,
            straight_through=straight_through,
            uniform_topk_eps=uniform_topk_eps,
        )  # alpha: (B, L, E)

        dt = (alpha @ torch.exp(self.log_dt)).transpose(-1, -2)
        B = (alpha @ self.B).transpose(-1, -2)  # (B, N, L)
        C = (alpha @ self.C).transpose(-1, -2)  # (B, N, L)

        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if self.d_conv > 0:
            if causal_conv1d_update is None:
                conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
                conv_state[:, :, -1] = x
                x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
                if self.conv1d.bias is not None:
                    x = x + self.conv1d.bias
                x = self.act(x).to(dtype=dtype)
            else:
                x = causal_conv1d_update(
                    x,
                    conv_state,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

        alpha, _ = self.router(
            x,
            noise_scale=0.0,
            mixture_temperature=1.0,
            straight_through=True,
            uniform_topk_eps=0.0,
        )  # alpha: (B, E)

        dt = alpha @ torch.exp(self.log_dt)
        A = -torch.exp(self.A_log.float())
        B = (alpha @ self.B)  # (B, N)
        C = (alpha @ self.C)  # (B, N)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class FantasticBlock(nn.Module):
    """Один блок: RMSNorm → Mamba → residual"""
    def __init__(
        self,
        d_model,
        num_experts=8,
        top_k=1,
        d_state=16,
        d_conv=4,
        expand=2,
        lb_strategy: str = "none",
        lb_coef: float = 0.01,
        aux_free_bias_step: float = 1e-3,
        dt_strategy: str = "random",
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fantastic = Fantastic(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            lb_strategy=lb_strategy,
            lb_coef=lb_coef,
            aux_free_bias_step=aux_free_bias_step,
        )

    def forward(self, x):
        return x + self.fantastic(self.norm(x))


class FantasticSSM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        num_experts: int = 8,
        top_k: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        lb_strategy: str = "none",
        lb_coef: float = 0.01,
        aux_free_bias_step: float = 1e-3,
        dt_strategy: str = "random",
        pad_vocab_size_multiple: int = 8,
    ):
        super().__init__()

        # Выравниваем vocab_size до кратного 8 (для эффективности)
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            FantasticBlock(
                d_model,
                num_experts,
                top_k,
                d_state,
                d_conv,
                expand,
                lb_strategy,
                lb_coef,
                aux_free_bias_step,
            ) for _ in range(n_layers)
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