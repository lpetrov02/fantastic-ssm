"""
DDP training script for Mamba130M / FantasticSSM.

Launch with torchrun:
    torchrun --standalone --nproc_per_node=NUM_GPUS train/train.py [args]

Example:
    torchrun --standalone --nproc_per_node=4 train/train.py \
        --model fantastic \
        --data_dir ./data/train \
        --val_dir  ./data/val  \
        --batch_size 8         \
        --grad_accum_steps 4   \
        --max_steps 100000
"""

import contextlib
import math
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Make project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mamba.mamba import Mamba130M
from models.fantastic.fantastic_ssm import FantasticSSM
from models.fantastic.fantastic_v0 import Fantastic_v0_SSM
from models.fantastic.fantastic_v2 import Fantastic_v2_SSM
from models.attentive.attentive import FantasticAttentiveSSM
from models.s4.s4d import S4DLanguageModel
from models.attention.mha import Transformer130M
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2Tokenizer
from train_data.data_loader import ShardedDataLoader, ValDataLoader
from train_data.niah_dataset import NIAHDataset, collate_niah, TRAIN_BASE_SEED

# Separate seed for NIAH validation — distinct from train (1337) and eval (42).
_NIAH_VAL_SEED = 2025


# ── CLI ───────────────────────────────────────────────────────────────────────

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train Mamba / Fantastic with DDP")

    # paths
    p.add_argument("--data_dir",       default="./data/train")
    p.add_argument("--val_dir",        default="./data/val")
    p.add_argument("--checkpoint_dir", default="./experiments/checkpoints")

    # model
    p.add_argument("--model",       choices=["mamba", "fantastic", "fantastic_v0", "fantastic_v2", "attentive", "s4d", "transformer"], default="mamba")
    p.add_argument("--model_name",  type=str)
    p.add_argument("--vocab_size",  type=int,   default=50257)
    p.add_argument("--d_model",     type=int,   default=768)
    p.add_argument("--n_layers",    type=int,   default=24)
    p.add_argument("--d_state",     type=int,   default=16)
    p.add_argument("--d_conv",      type=int,   default=4)
    p.add_argument("--expand",      type=int,   default=2)

    p.add_argument("--num_experts",        type=int,   default=8,   help="Fantastic only")
    p.add_argument("--top_k",              type=int,   default=1,   help="Fantastic only")
    p.add_argument("--lb_strategy",        type=str, default="none", help="Fantastic only")
    p.add_argument("--lb_coef",            type=float,   default=0.01,   help="Fantastic only")
    p.add_argument("--aux_free_bias_step", type=float,   default=0.001,   help="Fantastic only")
    p.add_argument("--dt_strategy", type=str,   default="random",   help="Fantastic only")
    p.add_argument("--basis_mode", action="store_true", help="Fantastic_mode")
    p.add_argument("--orthogonal_loss_coef", type=float, default=0.0, help="Fantastic only")
    p.add_argument("--separate_routing", action="store_true", help="Fantastic_mode")

    p.add_argument("--dt_rank",        type=str,   default="auto",   help="Fantastic-v2 only")
    p.add_argument("--dt_num_experts",        type=str,   default="auto",   help="Fantastic-v2 & Attentive")
    p.add_argument("--dt_top_k",              type=str,   default="auto",   help="Fantastic-v2 & Attentive")
    p.add_argument("--d_intermediate",        type=str,   default="auto",   help="Fantastic-Attentive only")
    p.add_argument("--orthogonal_loss_coef_dt", type=float, default=0.0, help="Fantastic-Attentive only")

    p.add_argument("--dropout",       type=float,   default=0.0,   help="S4DLanguageModel only")
    p.add_argument("--ff_mult",       type=int,   default=2,   help="FFN multiplier (S4DLanguageModel default=2; use 4 for transformer)")
    p.add_argument("--num_heads",     type=int,   default=12,  help="Number of attention heads (transformer only)")

    # training
    p.add_argument("--seq_len",          type=int,   default=2048)
    p.add_argument("--batch_size",       type=int,   default=16,    help="per-GPU batch size")
    p.add_argument("--grad_accum_steps", type=int,   default=1)
    p.add_argument("--max_steps",        type=int,   default=100_000)
    p.add_argument("--warmup_steps",     type=int,   default=2000)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--min_lr",           type=float, default=3e-5)
    p.add_argument("--weight_decay",     type=float, default=0.1)
    p.add_argument("--grad_clip",        type=float, default=1.0)

    # logging / checkpointing
    p.add_argument("--log_every",  type=int, default=10)
    p.add_argument("--val_every",  type=int, default=500)
    p.add_argument("--max_val_batches",  type=int, default=None)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--keep_ckpts", type=int, default=3, help="how many checkpoints to keep")
    p.add_argument("--train_offset", type=int, default=0)

    # embeddings
    p.add_argument("--use_pretrained", action="store_true", default=False,
                   help="Init embedding layer from GPT-2 weights (requires matching vocab_size and d_model)")
    p.add_argument("--freeze_embeds", action="store_true", default=False,
                   help="Freeze embedding weights (only effective with --use_pretrained)")

    # resume
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    p.add_argument("--resume_weights_only", action="store_true", default=False,
                   help="Load only model weights from --resume, not optimizer state. "
                        "Use this when starting NIAH fine-tuning from a pretrained checkpoint.")

    # NIAH fine-tuning
    p.add_argument("--niah", action="store_true", default=False,
                   help="Fine-tune on NIAH (passkey retrieval) instead of language modelling")
    p.add_argument("--niah_context_lengths", nargs="+", type=int, default=[2048],
                   help="Context lengths to include in NIAH training samples")
    p.add_argument("--niah_n_depths", type=int, default=9,
                   help="Number of evenly-spaced needle depths per context length")
    p.add_argument("--niah_n_samples", type=int, default=50,
                   help="Repeats per (context_length, depth) cell in training set")
    p.add_argument("--niah_batch_size", type=int, default=None,
                   help="Batch size for NIAH (defaults to --batch_size if unset)")

    args = p.parse_args()
    checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name)
    ensure_dir(checkpoint_path)
    args.checkpoint_dir = checkpoint_path

    return args


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(args) -> nn.Module:
    if args.model == "mamba":
        return Mamba130M(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )
    elif args.model == "fantastic":
        return FantasticSSM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            lb_strategy=args.lb_strategy,
            lb_coef=args.lb_coef,
            aux_free_bias_step=args.aux_free_bias_step,
            dt_strategy=args.dt_strategy,
            basis_mode=args.basis_mode,
            orthogonal_loss_coef=args.orthogonal_loss_coef,
            separate_routing=args.separate_routing,
        )
    elif args.model == "fantastic_v2":
        return Fantastic_v2_SSM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            lb_strategy=args.lb_strategy,
            lb_coef=args.lb_coef,
            aux_free_bias_step=args.aux_free_bias_step,
            dt_strategy=args.dt_strategy,
            basis_mode=args.basis_mode,
            orthogonal_loss_coef=args.orthogonal_loss_coef,
            dt_rank=args.dt_rank,
            dt_num_experts=args.dt_num_experts,
            dt_top_k=args.dt_top_k,
        )
    elif args.model == "attentive":
        return FantasticAttentiveSSM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_intermediate=args.d_intermediate,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            basis_mode=args.basis_mode,
            orthogonal_loss_coef_state=args.orthogonal_loss_coef,
            orthogonal_loss_coef_dt=args.orthogonal_loss_coef_dt,
            state_num_experts=args.num_experts,
            state_top_k=args.top_k,
            dt_num_experts=args.dt_num_experts,
            dt_top_k=args.dt_top_k,
        )
    elif args.model == "fantastic_v0":
        return Fantastic_v0_SSM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            d_state=args.d_state,
            lb_strategy=args.lb_strategy,
            lb_coef=args.lb_coef,
            aux_free_bias_step=args.aux_free_bias_step,
            dt_strategy=args.dt_strategy,
            orthogonal_loss_coef=args.orthogonal_loss_coef,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
        )
    elif args.model == "s4d":
        return S4DLanguageModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
        )
    elif args.model == "transformer":
        return Transformer130M(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            ffn_mult=args.ff_mult,
        )
    else:
        raise ValueError("Invalid model type")


# ── Pretrained embeddings ─────────────────────────────────────────────────────

def apply_pretrained_embeddings(model: nn.Module, args, is_main: bool):
    if not args.use_pretrained:
        return
    try:
        from transformers import GPT2Model
    except ImportError:
        raise ImportError("transformers package is required for --use_pretrained")

    gpt2 = GPT2Model.from_pretrained("gpt2")
    gpt2_emb = gpt2.wte.weight.data  # (50257, 768)

    if gpt2_emb.shape != (args.vocab_size, args.d_model):
        raise ValueError(
            f"GPT-2 embedding shape {tuple(gpt2_emb.shape)} does not match "
            f"vocab_size={args.vocab_size}, d_model={args.d_model}"
        )

    with torch.no_grad():
        num_embeds = gpt2_emb.shape[0]
        model.embedding.weight.data[:num_embeds].copy_(gpt2_emb)

    if args.freeze_embeds:
        model.embedding.weight.requires_grad_(False)
        if is_main:
            print("Embeddings loaded from GPT-2 and frozen.")
    else:
        if is_main:
            print("Embeddings loaded from GPT-2 (trainable).")


# ── Optimizer ─────────────────────────────────────────────────────────────────

def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    # Skip weight decay for biases and 1-D params (norms, embedding scales, …)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    if step >= args.max_steps:
        return args.min_lr
    progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Auxiliary loss (Fantastic routers) ────────────────────────────────────

def collect_aux_loss(raw_model: nn.Module):
    total = None
    for module in raw_model.modules():
        if hasattr(module, "get_auxiliary_loss"):
            loss = module.get_auxiliary_loss()
            if loss is not None:
                total = loss if total is None else total + loss
    return total  # None for plain Mamba


def collect_router_entropies(raw_model: nn.Module) -> list:
    # Returns per-layer entropy floats; empty list for models without routers.
    entropies = []
    for module in raw_model.modules():
        if hasattr(module, "get_entropy"):
            e = module.get_entropy()
            if e is not None:
                entropies.append(e)
    return entropies


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(step: int, raw_model: nn.Module, optimizer, args) -> Path:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )
    # Trim old checkpoints
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in ckpts[: -args.keep_ckpts]:
        old.unlink()
    return path


def load_checkpoint(path: str, raw_model: nn.Module, optimizer, device,
                    weights_only: bool = False) -> int:
    ckpt = torch.load(path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    raw_model.load_state_dict(state, strict=True)
    if not weights_only:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(raw_model: nn.Module, val_loader: ValDataLoader, device, max_batches=None) -> float:
    raw_model.eval()
    total_loss, n = 0.0, 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = raw_model(x)
        total_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        n += 1
        if max_batches and n >= max_batches:
            break
    raw_model.train()
    return total_loss / n if n > 0 else float("nan")


@torch.no_grad()
def _validate_niah(raw_model: nn.Module, val_loader, device) -> float:
    raw_model.eval()
    total_loss, n = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = raw_model(x)
        total_loss += F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
        ).item()
        n += 1
    raw_model.train()
    return total_loss / n if n > 0 else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    # ── Model + DDP ───────────────────────────────────────────────────
    raw_model = build_model(args).to(device)
    apply_pretrained_embeddings(raw_model, args, is_main)
    model = DDP(raw_model, device_ids=[local_rank])
    if is_main:
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"Model: {args.model} | params: {n_params / 1e6:.1f}M | GPUs: {world_size}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = make_optimizer(raw_model, args)

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume, raw_model, optimizer, device,
            weights_only=args.resume_weights_only,
        )
        if is_main:
            mode = "weights only" if args.resume_weights_only else "full"
            print(f"Resumed from {args.resume} at step {start_step} ({mode})")
        if args.resume_weights_only:
            start_step = 0  # fresh fine-tune run — reset step counter

    # ── Data ──────────────────────────────────────────────────────────
    if args.niah:
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        niah_bs = args.niah_batch_size or args.batch_size

        train_ds = NIAHDataset(
            tokenizer=tok,
            n_samples=args.niah_n_samples,
            context_lengths=args.niah_context_lengths,
            n_depths=args.niah_n_depths,
            base_seed=TRAIN_BASE_SEED,
        )
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=niah_bs, sampler=train_sampler,
            collate_fn=collate_niah, drop_last=True,
        )

        if is_main:
            val_ds = NIAHDataset(
                tokenizer=tok,
                n_samples=20,
                context_lengths=args.niah_context_lengths,
                n_depths=args.niah_n_depths,
                base_seed=_NIAH_VAL_SEED,
            )
            val_loader = DataLoader(
                val_ds, batch_size=niah_bs, collate_fn=collate_niah
            )
        else:
            val_loader = None

        if is_main:
            print(
                f"NIAH mode | train={len(train_ds)} samples | "
                f"val={len(val_ds)} samples | batch={niah_bs}"
            )
    else:
        train_loader = ShardedDataLoader(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            seed=42,
            offset=args.train_offset,
        )
        # Validation runs on rank-0 only (ValDataLoader is not DDP-aware)
        val_loader = ValDataLoader(
            data_dir=args.val_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        ) if is_main else None

    # ── TensorBoard (rank 0 only) ──────────────────────────────────────
    writer = None
    if is_main:
        tb_dir = Path(args.checkpoint_dir) / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    step = start_step

    if args.niah:
        # DistributedSampler requires set_epoch each pass for proper shuffling.
        def infinite_batches():
            epoch = 0
            while True:
                train_sampler.set_epoch(epoch)
                yield from train_loader
                epoch += 1
    else:
        tokens_per_log = (
            args.batch_size * world_size * args.seq_len
            * args.grad_accum_steps * args.log_every
        )
        def infinite_batches():
            while True:
                yield from train_loader

    data = infinite_batches()
    t0 = time.perf_counter()

    while step < args.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(args.grad_accum_steps):
            x, y = next(data)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Skip all-reduce on every micro-step except the last
            sync_ctx = (
                contextlib.nullcontext()
                if micro == args.grad_accum_steps - 1
                else model.no_sync()
            )
            with sync_ctx:
                logits = model(x)
                if args.niah:
                    # y has -100 at context positions; loss on answer tokens only
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
                    )
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.grad_accum_steps
                aux = collect_aux_loss(raw_model)
                if aux is not None:
                    total_loss = loss + aux / args.grad_accum_steps
                else:
                    total_loss = loss
                total_loss.backward()

            accum_loss += loss.item()

        # LR schedule
        lr = get_lr(step, args)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────
        if is_main and step % args.log_every == 0:
            t1 = time.perf_counter()

            entropies = collect_router_entropies(raw_model)
            entropy_str = ""
            if entropies:
                mean_h = sum(entropies) / len(entropies)
                entropy_str = f" | H {mean_h:.3f}"
                writer.add_scalar("router/entropy/mean", mean_h,          step)
                writer.add_scalar("router/entropy/min",  min(entropies),   step)
                writer.add_scalar("router/entropy/max",  max(entropies),   step)
                for i, h in enumerate(entropies):
                    writer.add_scalar(f"router/entropy/layer_{i}", h, step)

            if args.niah:
                elapsed = t1 - t0
                print(
                    f"step {step:7d} | niah_loss {accum_loss:.4f} | lr {lr:.2e}"
                    f" | {elapsed:.1f}s{entropy_str}"
                )
            else:
                tok_per_sec = tokens_per_log / (t1 - t0)
                print(
                    f"step {step:7d} | loss {accum_loss:.4f} | lr {lr:.2e}"
                    f" | {tok_per_sec / 1e3:.1f}k tok/s{entropy_str}"
                )
                writer.add_scalar("train/tokens_per_sec", tok_per_sec, step)

            writer.add_scalar("train/loss", accum_loss, step)
            writer.add_scalar("train/lr",   lr,         step)
            t0 = t1

        # ── Validation ────────────────────────────────────────────────
        if is_main and step > 0 and step % args.val_every == 0:
            if args.niah:
                val_loss = _validate_niah(raw_model, val_loader, device)
            else:
                val_loss = validate(raw_model, val_loader, device, max_batches=args.max_val_batches)
            print(f"  val  loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}")
            writer.add_scalar("val/loss", val_loss,           step)
            writer.add_scalar("val/ppl",  math.exp(val_loss), step)

        # ── Checkpoint ────────────────────────────────────────────────
        if is_main and step > 0 and step % args.save_every == 0:
            ckpt_path = save_checkpoint(step, raw_model, optimizer, args)
            print(f"  ckpt → {ckpt_path}")

        step += 1

    # Final save
    if is_main:
        ckpt_path = save_checkpoint(step, raw_model, optimizer, args)
        print(f"Done. Final checkpoint → {ckpt_path}")
        writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
