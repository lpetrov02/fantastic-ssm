# Evaluations

## Needle-in-a-Haystack (NIAH)

Passkey-retrieval NIAH benchmark for all `fantastic-ssm` model variants.

## How it works

Each sample has the form:

```
[haystack before] [needle] [haystack after] [query] [answer]
```

- **Haystack** — repeated filler sentences (e.g. "The sky is blue.")
- **Needle** — `"The pass key is: 73921. Remember it. 73921 is the pass key."`
- **Query** — `"\nWhat is the pass key? The pass key is:"`
- **Answer** — `" 73921"`

The model's cross-entropy loss is measured **only on the answer tokens**. Lower loss means the model successfully retrieved the needle from context.

The evaluation sweeps a grid of **context lengths × needle depths**, running `n_samples` samples per cell and averaging the loss.

## Files

| File | Purpose |
|---|---|
| `niah_data.py` | Sample generation — haystack construction, needle insertion, tokenization |
| `niah_eval.py` | Main evaluation script — loads checkpoint, runs grid, saves JSON |
| `plot_results.py` | Renders a heatmap from one or more JSON result files |
| `compare_models.py` | Wrapper that evaluates multiple checkpoints then plots them side-by-side |
| `run_niah.sh` | Shell script for a quick single-model run |

Results are saved under `evaluation/results/`.

---

## Tiny-Shakespeare Perplexity

Standard perplexity benchmark on the Tiny-Shakespeare corpus. The script downloads the dataset once (cached to `evaluation/data/shakespeare.txt`), tokenises it with the GPT-2 tokeniser, and evaluates the model on the last `test_fraction` of the text.

### Quick start

```bash
python evaluation/shakespeare_eval.py \
    --checkpoint checkpoints/step_10000.pt \
    --output     evaluation/results/shakespeare_fantastic.json
```

### `shakespeare_eval.py` options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to `.pt` checkpoint |
| `--output` | `evaluation/results/shakespeare.json` | Output JSON path |
| `--seq_len` | `1024` | Sequence length for each evaluation chunk |
| `--test_fraction` | `0.1` | Fraction of dataset used as test split (last N%) |
| `--batch_size` | `4` | Chunks per forward pass |
| `--device` | `cuda` / `cpu` | Inference device |
| `--seed` | `42` | Random seed |

### Output format

```json
{
  "checkpoint": "checkpoints/step_10000.pt",
  "model": "fantastic",
  "model_step": 10000,
  "eval_config": {
    "seq_len": 1024,
    "test_fraction": 0.1,
    "seed": 42
  },
  "results": {
    "mean_loss": 3.21,
    "mean_ppl": 24.78,
    "n_chunks": 42,
    "per_chunk_losses": [...],
    "quartiles": {
      "q1": {"mean_loss": 3.10, "mean_ppl": 22.20},
      "q2": {"mean_loss": 3.18, "mean_ppl": 24.05},
      "q3": {"mean_loss": 3.25, "mean_ppl": 25.79},
      "q4": {"mean_loss": 3.31, "mean_ppl": 27.38}
    }
  }
}
```

## Quick start

### Single model

```bash
python evaluation/niah_eval.py \
    --checkpoint checkpoints/step_10000.pt \
    --output     evaluation/results/niah_fantastic.json

python evaluation/plot_results.py evaluation/results/niah_fantastic.json
```

### Compare two models

```bash
python evaluation/compare_models.py \
    --checkpoints checkpoints/fantastic.pt checkpoints/mamba.pt \
    --labels      Fantastic Mamba
```

Or using the shell script (pass checkpoint as first argument):

```bash
bash evaluation/run_niah.sh checkpoints/step_10000.pt
```

## `niah_eval.py` options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to `.pt` checkpoint |
| `--output` | `evaluation/results/niah.json` | Output JSON path |
| `--context_lengths` | `512 1024 2048` | Context lengths to sweep |
| `--n_depths` | `9` | Number of evenly-spaced depth positions |
| `--n_samples` | `10` | Samples per (context\_length, depth) cell |
| `--batch_size` | `4` | Samples per forward pass |
| `--device` | `cuda` / `cpu` | Inference device |
| `--seed` | `42` | Random seed |

## `plot_results.py` options

| Flag | Default | Description |
|---|---|---|
| `results` | *(required)* | One or more JSON result files |
| `--metric` | `loss` | `loss` or `ppl` (perplexity) |
| `--labels` | model name from JSON | Override display labels |
| `--out` | auto | Output image path |

## Output format

`niah_eval.py` saves a JSON file with this structure:

```json
{
  "checkpoint": "checkpoints/step_10000.pt",
  "model": "fantastic",
  "model_step": 10000,
  "eval_config": {
    "context_lengths": [512, 1024, 2048],
    "depth_pcts": [0.1111, 0.2222, ...],
    "n_samples": 10,
    "seed": 42
  },
  "results": {
    "512": {
      "0.1111": {
        "mean_loss": 1.23,
        "mean_ppl": 3.42,
        "per_sample_losses": [1.21, 1.25, ...]
      }
    }
  }
}
```

## Interpreting results

Results are visualised as a heatmap (rows = context lengths, columns = needle depths). Color scale runs **green → red** (low loss → high loss).

- A model with good long-range memory should have **uniformly low loss** across all depths and context lengths.
- Loss that degrades at large context lengths or at needle positions far from the query indicates the model fails to carry information over long distances.
- SSM-based models (Mamba, Fantastic) do not have explicit attention but use state compression, so performance at long contexts and early needle depths is the most informative test.
