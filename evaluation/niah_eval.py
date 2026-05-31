"""
Needle-in-a-Haystack evaluation for fantastic-ssm models.

Usage:
    python evaluation/niah_eval.py \\
        --checkpoint checkpoints/step_10000.pt \\
        --output    evaluation/results/niah_fantastic.json \\
        [--context_lengths 512 1024 2048] \\
        [--n_depths 9] \\
        [--n_samples 10] \\
        [--batch_size 4] \\
        [--device cuda]

The script loads a checkpoint (which stores the model args), instantiates
the right model class, and runs the passkey-retrieval NIAH evaluation.
Loss is measured only on the passkey answer tokens; lower is better.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import GPT2Tokenizer

from models.mamba.mamba import Mamba130M
from models.fantastic.fantastic_ssm import FantasticSSM
from models.fantastic.fantastic_v0 import Fantastic_v0_SSM
from models.fantastic.fantastic_v2 import Fantastic_v2_SSM
from models.attentive.attentive import FantasticAttentiveSSM
from models.s4.s4d import S4DLanguageModel

from evaluation.niah_data import build_grid, NIAHSample


# ── model loading ──────────────────────────────────────────────────────────────

def build_model(args: dict):
    """Instantiate the right model class from a checkpoint's saved args dict."""
    model_type = args.get("model", "fantastic")

    common = dict(
        vocab_size=args.get("vocab_size", 50257),
        d_model=args.get("d_model", 768),
        n_layers=args.get("n_layers", 24),
    )

    if model_type == "mamba":
        return Mamba130M(**common)

    if model_type == "attentive":
        return FantasticAttentiveSSM(
            **common,
            d_intermediate=args.get("d_intermediate", "auto"),
            d_state=args.get("d_state", 16),
            d_conv=args.get("d_conv", 4),
            expand=args.get("expand", 2),
            basis_mode=args.get("basis_mode", False),
            orthogonal_loss_coef_state=args.get("orthogonal_loss_coef", 0.0),
            orthogonal_loss_coef_dt=args.get("orthogonal_loss_coef_dt", 0.0),
            state_num_experts=args.get("num_experts", "auto"),
            state_top_k=args.get("top_k", "auto"),
            dt_num_experts=args.get("dt_num_experts", "auto"),
            dt_top_k=args.get("dt_top_k", "auto"),
        )

    if model_type in ("fantastic", "fantastic_v0", "fantastic_v2"):
        fantastic_kwargs = dict(
            **common,
            d_state=args.get("d_state", 16),
            d_conv=args.get("d_conv", 4),
            expand=args.get("expand", 2),
            num_experts=args.get("num_experts", 8),
            top_k=args.get("top_k", 1),
            dt_strategy=args.get("dt_strategy", "random"),
            basis_mode=args.get("basis_mode", False),
            orthogonal_loss_coef=args.get("orthogonal_loss_coef", 0.0),
            lb_strategy=args.get("lb_strategy", "none"),
            lb_coef=args.get("lb_coef", 0.01),
            separate_routing=args.get("separate_routing", False),
            dt_rank=args.get("dt_rank", "auto"),
            dt_num_experts=args.get("dt_num_experts", "auto"),
            dt_top_k=args.get("dt_top_k", "auto"),
        )
        cls = FantasticSSM if model_type == "fantastic" else (Fantastic_v2_SSM if model_type == "fantastic_v2" else Fantastic_v0_SSM)
        return cls(**fantastic_kwargs)

    if model_type == "s4d":
        return S4DLanguageModel(
            **common,
            d_state=args.get("d_state", 16),
            dropout=args.get("dropout", 0.0),
            ff_mult=args.get("ff_mult", 2),
        )

    raise ValueError(f"Unknown model type: {model_type!r}")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    saved_args = ckpt.get("args", {})
    print("Model ARGS:")
    print(saved_args)
    model = build_model(saved_args)
    # Strip DDP wrapper prefix if present
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    # print(state.keys())
    model.load_state_dict(state, strict=True)
    model.to(device)
    # model.eval()
    model.train()
    print(
        f"Loaded {saved_args.get('model', '?')} checkpoint "
        f"(step {ckpt.get('step', '?')}) from {path}"
    )
    return model, saved_args


# ── evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_batch(model, samples: List[NIAHSample], device: torch.device) -> List[float]:
    """
    Run a batch of NIAH samples through the model.
    Returns per-sample mean cross-entropy loss on the answer tokens.
    """
    max_len = max(s.answer_end for s in samples)
    batch_size = len(samples)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, s in enumerate(samples):
        t = torch.tensor(s.input_ids, dtype=torch.long)
        input_ids[i, : len(s.input_ids)] = t

    # Forward — models return (B, L, vocab_size)
    logits = model(input_ids)

    losses = []
    for i, s in enumerate(samples):
        # Predict positions [answer_start .. answer_end-1] from context tokens
        pred_logits = logits[i, s.answer_start - 1 : s.answer_end - 1]   # (A, V)
        targets = input_ids[i, s.answer_start : s.answer_end]             # (A,)
        # print(F.cross_entropy(pred_logits, targets, reduction="none"))
        loss = F.cross_entropy(pred_logits, targets).item()
        losses.append(loss)

    return losses


def run_evaluation(
    model,
    tokenizer,
    context_lengths: List[int],
    depth_pcts: List[float],
    n_samples: int,
    batch_size: int,
    device: torch.device,
    make_random: bool = False,
    shots: int = 0,
) -> dict:
    """
    Evaluate on the full NIAH grid.
    Returns nested dict: results[ctx_len][depth_pct] = {"mean_loss": float, "mean_ppl": float}
    """
    print("Building sample grid …")
    grid = build_grid(tokenizer, context_lengths, depth_pcts, n_samples=n_samples, make_random=make_random, shots=shots)

    results = {}
    total_cells = len(context_lengths) * len(depth_pcts)
    cell_idx = 0

    for ctx_len in context_lengths:
        results[ctx_len] = {}
        for depth in depth_pcts:
            cell_idx += 1
            samples = grid[ctx_len][depth]

            all_losses = []
            for start in range(0, len(samples), batch_size):
                chunk = samples[start : start + batch_size]
                all_losses.extend(eval_batch(model, chunk, device))

            mean_loss = sum(all_losses) / len(all_losses)
            mean_ppl = math.exp(mean_loss)
            results[ctx_len][depth] = {
                "mean_loss": round(mean_loss, 4),
                "mean_ppl": round(mean_ppl, 4),
                "per_sample_losses": [round(l, 4) for l in all_losses],
            }
            print(
                f"  [{cell_idx:3d}/{total_cells}] "
                f"ctx={ctx_len:5d}  depth={depth:.0%}  "
                f"loss={mean_loss:.3f}  ppl={mean_ppl:.2f}"
            )

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NIAH evaluation for fantastic-ssm")
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint .pt file")
    p.add_argument("--output", default="evaluation/results/niah.json",
                   help="Where to save JSON results")
    p.add_argument("--context_lengths", nargs="+", type=int,
                   default=[512, 1024, 2048],
                   help="Context lengths to evaluate (max = model seq_len)")
    p.add_argument("--n_depths", type=int, default=9,
                   help="Number of evenly-spaced depth positions to test")
    p.add_argument("--n_samples", type=int, default=10,
                   help="Samples per (context_length, depth) cell")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Samples per forward pass")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--make_random", action="store_true", default=False)
    p.add_argument("--shots", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Depth positions: n evenly-spaced points in (0, 1]
    depth_pcts = [round((i + 1) / args.n_depths, 4) for i in range(args.n_depths)]

    print(f"Device : {device}")
    print(f"Context lengths : {args.context_lengths}")
    print(f"Depth positions : {[f'{d:.0%}' for d in depth_pcts]}")
    print(f"Samples per cell: {args.n_samples}")
    print()

    model, saved_args = load_checkpoint(args.checkpoint, device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        context_lengths=args.context_lengths,
        depth_pcts=depth_pcts,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
        make_random=args.make_random,
        shots=args.shots,
    )

    output = {
        "checkpoint": args.checkpoint,
        "model": saved_args.get("model", "unknown"),
        "model_step": None,
        "eval_config": {
            "context_lengths": args.context_lengths,
            "depth_pcts": depth_pcts,
            "n_samples": args.n_samples,
            "seed": args.seed,
        },
        "results": {
            str(ctx): {str(d): v for d, v in depths.items()}
            for ctx, depths in results.items()
        },
    }

    # Recover step from checkpoint filename as fallback
    try:
        ckpt_stem = Path(args.checkpoint).stem
        for part in ckpt_stem.split("_"):
            if part.isdigit():
                output["model_step"] = int(part)
                break
    except Exception:
        pass

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
