"""
Needle-in-a-Haystack data generation.

Uses the passkey retrieval format:
  [haystack_before] [needle] [haystack_after] [query] [answer]

Loss is measured only on answer tokens.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


HAYSTACK_SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is shining.",
    "The sea is nice.",
    "The birds are singing.",
    "The flowers are blooming.",
    "The trees are tall.",
    "The mountains are high.",
    "The rivers are flowing.",
    "The clouds are white.",
    "The weather is pleasant.",
    "The wind is calm.",
    "The leaves are falling.",
    "The stars are bright.",
    "The moon is glowing.",
]

NEEDLE_TEMPLATE = (
    "The pass key is: {passkey}. Remember it. {passkey} is the pass key."
)
QUERY_PREFIX = "\nWhat is the pass key? The pass key is:"


@dataclass
class NIAHSample:
    input_ids: List[int]
    answer_start: int      # index of first answer token in input_ids
    answer_end: int        # exclusive
    passkey: str
    context_length: int    # tokens before answer (haystack + needle + query)
    depth_pct: float


def _random_passkey() -> str:
    return str(random.randint(10000, 99999))


def _build_haystack(tokenizer, n_tokens: int, rng: random.Random) -> List[int]:
    ids: List[int] = []
    while len(ids) < n_tokens:
        sentence = rng.choice(HAYSTACK_SENTENCES)
        ids.extend(tokenizer.encode(" " + sentence))
    return ids[:n_tokens]


def build_sample(
    tokenizer,
    context_length: int,
    depth_pct: float,
    passkey: Optional[str] = None,
    seed: Optional[int] = None,
    make_random: bool = False,
    shots: int = 0,
) -> NIAHSample:
    """
    Build one NIAH sample.

    context_length  -- total tokens in [haystack + needle + query]
    depth_pct       -- where in the haystack [0.0, 1.0] to insert the needle
    """

    prefix = []
    if shots > 0:
        for _ in range(shots):
            depth = np.random.choice(np.linspace(0.1, 1, 5))
            prefix.append(build_sample(tokenizer, 64, depth, shots=0))

    rng = random.Random(seed)

    if passkey is None:
        passkey = _random_passkey()

    needle_ids = tokenizer.encode(NEEDLE_TEMPLATE.format(passkey=passkey))
    query_ids = tokenizer.encode(QUERY_PREFIX)

    final_passkey = _random_passkey() if make_random else passkey
    answer_ids = tokenizer.encode(" " + final_passkey)

    haystack_budget = context_length - len(needle_ids) - len(query_ids)
    if haystack_budget < 4:
        raise ValueError(
            f"context_length={context_length} is too short for needle "
            f"({len(needle_ids)} tok) + query ({len(query_ids)} tok)."
        )

    haystack_ids = _build_haystack(tokenizer, haystack_budget, rng)

    split = int(len(haystack_ids) * depth_pct)
    before = haystack_ids[:split]
    after = haystack_ids[split:]

    context_ids = before + needle_ids + after + query_ids
    # Pad / trim to exact context_length
    context_ids = context_ids[:context_length]

    full_ids = context_ids + answer_ids
    answer_start = len(context_ids)

    # print(tokenizer.decode(full_ids))

    if prefix:
        prefix_ids = tokenizer.encode("Examples:\n")
        for sample in prefix:
            prefix_ids += sample.input_ids
            prefix_ids += tokenizer.encode("\n###\n")
        prefix_ids += tokenizer.encode("Now your turn:\n")
        full_ids = prefix_ids + full_ids
        answer_start += len(prefix_ids)
 
    # if shots > 0:
    #     print(tokenizer.decode(full_ids))
    #     raise ValueError()

    return NIAHSample(
        input_ids=full_ids,
        answer_start=answer_start,
        answer_end=len(full_ids),
        passkey=passkey,
        context_length=len(context_ids),
        depth_pct=depth_pct,
    )


def build_grid(
    tokenizer,
    context_lengths: List[int],
    depth_pcts: List[float],
    n_samples: int = 10,
    base_seed: int = 42,
    make_random: bool = False,
    shots: int = 0,
) -> dict:
    """
    Build all (context_length, depth_pct) × n_samples samples.

    Returns a nested dict:
        grid[context_length][depth_pct] = List[NIAHSample]
    """
    grid = {}
    for ctx_len in context_lengths:
        grid[ctx_len] = {}
        for depth in depth_pcts:
            samples = []
            for i in range(n_samples):
                seed = base_seed + hash((ctx_len, depth, i)) % (2 ** 31)
                samples.append(
                    build_sample(
                        tokenizer,
                        context_length=ctx_len,
                        depth_pct=depth,
                        seed=seed,
                        make_random=make_random,
                        shots=shots,
                    )
                )
            grid[ctx_len][depth] = samples
    return grid
