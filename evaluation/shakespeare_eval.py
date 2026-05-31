"""
Tiny-Shakespeare perplexity evaluation for fantastic-ssm models.

Usage:
    python evaluation/shakespeare_eval.py \\
        --checkpoint checkpoints/step_10000.pt \\
        --output     evaluation/results/shakespeare_fantastic.json \\
        [--seq_len 1024] \\
        [--test_fraction 0.1] \\
        [--batch_size 4] \\
        [--device cuda]

The script downloads Tiny-Shakespeare on first run (cached to evaluation/data/),
tokenises it with the GPT-2 tokeniser, and evaluates the model on the last
`test_fraction` of the text.  Loss is measured on every token; lower is better.
"""

import argparse
import json
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import GPT2Tokenizer

from models.mamba.mamba import Mamba130M
from models.fantastic.fantastic_ssm import FantasticSSM
from models.fantastic.fantastic_v0 import Fantastic_v0_SSM
from models.fantastic.fantastic_v2 import Fantastic_v2_SSM
from models.s4.s4d import S4DLanguageModel
from models.attentive.attentive import FantasticAttentiveSSM
from models.attention.mha import Transformer130M


_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
_CACHE_PATH = Path(__file__).resolve().parent / "data" / "shakespeare.txt"


# ── data ──────────────────────────────────────────────────────────────────────

def load_shakespeare(cache_path: Path = _CACHE_PATH) -> str:
    if not cache_path.exists():
        print(f"Downloading Tiny-Shakespeare to {cache_path} …")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_SHAKESPEARE_URL, cache_path)
    return cache_path.read_text(encoding="utf-8")


def tokenise(text: str, tokenizer) -> List[int]:
    return tokenizer.encode(text)


def make_chunks(token_ids: List[int], seq_len: int) -> List[List[int]]:
    """Split token_ids into non-overlapping windows of seq_len+1 (input + target)."""
    chunks = []
    window = seq_len + 1
    for start in range(0, len(token_ids) - window + 1, window):
        chunks.append(token_ids[start : start + window])
    return chunks


# ── model loading ──────────────────────────────────────────────────────────────

def build_model(args: dict):
    model_type = args.get("model", "fantastic")

    common = dict(
        vocab_size=args.get("vocab_size", 50257),
        d_model=args.get("d_model", 768),
        n_layers=args.get("n_layers", 24),
    )

    if model_type == "mamba":
        return Mamba130M(**common)

    if model_type in ("fantastic", "fantastic_v0", "fantastic_v2"):
        kwargs = dict(
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
        cls = (
            FantasticSSM if model_type == "fantastic"
            else Fantastic_v2_SSM if model_type == "fantastic_v2"
            else Fantastic_v0_SSM
        )
        return cls(**kwargs)

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

    if model_type == "s4d":
        return S4DLanguageModel(
            **common,
            d_state=args.get("d_state", 16),
            dropout=args.get("dropout", 0.0),
            ff_mult=args.get("ff_mult", 2),
        )

    if model_type == "transformer":
        return Transformer130M(
            **common,
            num_heads=args.get("num_heads", 12),
            ffn_mult=args.get("ff_mult", 4),
        )

    raise ValueError(f"Unknown model type: {model_type!r}")


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    saved_args = ckpt.get("args", {})
    print("Model args:", saved_args)
    model = build_model(saved_args)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.train()
    print(
        f"Loaded {saved_args.get('model', '?')} checkpoint "
        f"(step {ckpt.get('step', '?')}) from {path}"
    )
    return model, saved_args


# ── evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_chunks(
    model,
    chunks: List[List[int]],
    batch_size: int,
    device: torch.device,
) -> List[float]:
    """Return per-chunk mean cross-entropy loss (over all seq_len tokens)."""
    losses = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        B = len(batch)
        seq_len = len(batch[0]) - 1

        tensor = torch.tensor(batch, dtype=torch.long, device=device)  # (B, seq_len+1)
        x = tensor[:, :-1]   # (B, seq_len)
        y = tensor[:, 1:]    # (B, seq_len)

        logits = model(x)    # (B, seq_len, vocab_size)
        V = logits.size(-1)

        loss = F.cross_entropy(
            logits.reshape(B * seq_len, V),
            y.reshape(B * seq_len),
            reduction="none",
        ).reshape(B, seq_len).mean(dim=1)  # (B,)

        losses.extend(loss.tolist())
    return losses


def confidence_interval(losses: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Return (lower, upper) CI for the mean loss using a t-distribution."""
    n = len(losses)
    if n < 2:
        mean = losses[0] if losses else float("nan")
        return mean, mean
    mean = sum(losses) / n
    se = stats.sem(losses)
    h = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean - h, mean + h


def summarise_losses(losses: List[float], confidence: float = 0.95) -> dict:
    mean = sum(losses) / len(losses)
    lo, hi = confidence_interval(losses, confidence)
    return {
        "mean_loss": round(mean, 4),
        "mean_ppl": round(math.exp(mean), 4),
        "ci_loss": [round(lo, 4), round(hi, 4)],
        "ci_ppl": [round(math.exp(lo), 4), round(math.exp(hi), 4)],
    }


def run_evaluation(
    model,
    chunks: List[List[int]],
    batch_size: int,
    device: torch.device,
    confidence: float = 0.95,
) -> dict:
    n = len(chunks)
    print(f"Evaluating on {n} chunks …")

    all_losses = eval_chunks(model, chunks, batch_size, device)

    overall = summarise_losses(all_losses, confidence)

    # Per-quartile breakdown (useful for spotting positional degradation)
    q = max(1, n // 4)
    quartiles = {}
    for i, label in enumerate(["q1", "q2", "q3", "q4"]):
        chunk = all_losses[i * q : (i + 1) * q]
        if not chunk:
            break
        quartiles[label] = summarise_losses(chunk, confidence)

    return {
        **overall,
        "confidence": confidence,
        "n_chunks": n,
        "per_chunk_losses": [round(l, 4) for l in all_losses],
        "quartiles": quartiles,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tiny-Shakespeare perplexity eval")
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint .pt file")
    p.add_argument("--output", default="evaluation/results/shakespeare.json",
                   help="Where to save JSON results")
    p.add_argument("--seq_len", type=int, default=1024,
                   help="Sequence length for each evaluation chunk")
    p.add_argument("--test_fraction", type=float, default=0.1,
                   help="Fraction of the dataset to use as the test split (last N%%)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Chunks per forward pass")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence level for intervals (default: 0.95)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"Device        : {device}")
    print(f"Seq length    : {args.seq_len}")
    print(f"Test fraction : {args.test_fraction:.0%}")
    print()

    # Data
    text = load_shakespeare()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    all_ids = tokenise(text, tokenizer)

    split = int(len(all_ids) * (1.0 - args.test_fraction))
    test_ids = all_ids[split:]
    chunks = make_chunks(test_ids, args.seq_len)

    print(f"Total tokens  : {len(all_ids):,}")
    print(f"Test tokens   : {len(test_ids):,}")
    print(f"Chunks        : {len(chunks)}")
    print()

    model, saved_args = load_checkpoint(args.checkpoint, device)

    results = run_evaluation(model, chunks, args.batch_size, device, confidence=args.confidence)

    ci = results["confidence"]
    lo_l, hi_l = results["ci_loss"]
    lo_p, hi_p = results["ci_ppl"]
    print(
        f"\nOverall  loss: {results['mean_loss']:.4f}  [{lo_l:.4f}, {hi_l:.4f}]  ({ci:.0%} CI)"
        f"   ppl: {results['mean_ppl']:.2f}  [{lo_p:.2f}, {hi_p:.2f}]"
    )
    for qk, qv in results["quartiles"].items():
        ql_lo, ql_hi = qv["ci_loss"]
        qp_lo, qp_hi = qv["ci_ppl"]
        print(
            f"  {qk}: loss={qv['mean_loss']:.4f} [{ql_lo:.4f}, {ql_hi:.4f}]"
            f"  ppl={qv['mean_ppl']:.2f} [{qp_lo:.2f}, {qp_hi:.2f}]"
        )

    output = {
        "checkpoint": args.checkpoint,
        "model": saved_args.get("model", "unknown"),
        "model_step": None,
        "eval_config": {
            "seq_len": args.seq_len,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
        },
        "results": results,
    }

    try:
        for part in Path(args.checkpoint).stem.split("_"):
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
