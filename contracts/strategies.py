"""Three ways to get a routing decision out of the same language model.

1. free_decode           Generate text token by token, then parse the decision out of it.
                         Output contract: a string. Parsing can fail.
2. constrained_decode    Generate JSON token by token while a grammar masks every token that
                         would break the schema. Output contract: valid JSON, always.
3. score_candidates      Hand the model the allowed answers up front and score each one in a
                         single teacher-forced pass. Output contract: a probability for every
                         allowed answer. Nothing is generated.

Every function returns how many sequential forward passes it needed and how many
token positions it pushed through the model, so the inference cost is visible.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field

import numpy as np

from .data import JSON_TEMPLATE, LABELS
from .lm import EOS, TinyLM, tokenize


@dataclass
class Result:
    label: str | None                 # the decision the application gets (None = unusable output)
    raw: str                          # what the model produced (or the scored distribution)
    forward_passes: int
    positions: int
    probs: dict[str, float] = field(default_factory=dict)   # only scoring fills this

    @property
    def valid(self) -> bool:
        return self.label is not None


def _pick(logprobs: np.ndarray, temperature: float, rng: random.Random | None) -> int:
    if temperature <= 0:
        return int(np.argmax(logprobs))
    z = logprobs / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    r = (rng or random.Random()).random()
    return int(min(np.searchsorted(np.cumsum(p), r), len(p) - 1))


# ---------------------------------------------------------------- 1) free decoding

_LABEL_RE = re.compile("|".join(re.escape(l) for l in sorted(LABELS, key=len, reverse=True)))


def parse_label(text: str) -> str | None:
    """Recover the decision from prose: exactly one team must be named."""
    found = set(_LABEL_RE.findall(text))
    return found.pop() if len(found) == 1 else None


def detok_json(tokens: list[str]) -> str:
    """Rebuild JSON text from space-separated tokens (words inside strings keep their spaces)."""
    out, in_str, words = [], False, []
    for t in tokens:
        if t == '"':
            if in_str:
                out.append('"' + " ".join(words) + '"')
                words = []
            in_str = not in_str
        elif in_str:
            words.append(t)
        else:
            out.append(t)
    if in_str:                                   # unterminated string: keep it broken
        out.append('"' + " ".join(words))
    return "".join(out)


def free_decode(lm: TinyLM, ticket: str, style: str = "prose", max_tokens: int = 24,
                temperature: float = 0.0, rng: random.Random | None = None) -> Result:
    lm.reset_counters()
    toks: list[str] = []
    for _ in range(max_tokens):
        j = _pick(lm.next_logprobs(ticket, style, toks), temperature, rng)
        w = lm.vocab[j]
        if w == EOS:
            break
        toks.append(w)
    if style == "json":
        raw = detok_json(toks)
        try:
            label = json.loads(raw).get("team")
            label = label if label in LABELS else None
        except (json.JSONDecodeError, AttributeError):
            label = None
    else:
        raw = " ".join(toks)
        label = parse_label(raw)
    return Result(label, raw, lm.forward_passes, lm.positions)


# ---------------------------------------------------- 2) grammar-constrained decoding

def json_sequences(labels=LABELS) -> list[list[str]]:
    """Every token sequence the schema {"team": <one of labels>} allows."""
    return [tokenize(JSON_TEMPLATE.format(label=l)) + [EOS] for l in labels]


def constrained_decode(lm: TinyLM, ticket: str, labels=LABELS, fast_forward: bool = False,
                       temperature: float = 0.0, rng: random.Random | None = None) -> Result:
    """Decode JSON while masking any token the schema does not allow at this position.

    The grammar here is the set of complete valid outputs; at each step the allowed
    tokens are those that keep the prefix inside that set (a trie walk, which is
    what regex/CFG-guided decoders compile schemas into).

    fast_forward=True skips the model call when the grammar allows exactly one
    token (`{`, `"`, `team`, `:`...). Structured-generation engines do this to avoid
    paying a forward pass for tokens that are already determined.
    """
    lm.reset_counters()
    allowed_seqs = json_sequences(labels)
    toks: list[str] = []
    while True:
        live = [s for s in allowed_seqs if s[:len(toks)] == toks and len(s) > len(toks)]
        allowed = sorted({s[len(toks)] for s in live})
        if not allowed:
            break
        if fast_forward and len(allowed) == 1:
            w = allowed[0]
        else:
            lp = lm.next_logprobs(ticket, "json", toks)
            mask = np.full_like(lp, -np.inf)
            for a in allowed:
                mask[lm.token_id(a)] = 0.0
            w = lm.vocab[_pick(lp + mask, temperature, rng)]
        if w == EOS:
            break
        toks.append(w)
    raw = detok_json(toks)
    return Result(json.loads(raw)["team"], raw, lm.forward_passes, lm.positions)


# ------------------------------------------------------------- 3) candidate scoring

def score_candidates(lm: TinyLM, ticket: str, labels=LABELS, length_normalize: bool = False) -> Result:
    """Score every allowed answer and return a probability distribution over them.

    Each candidate is scored as log P(candidate tokens, <eos> | prompt). The
    softmax across candidates gives the distribution the application receives;
    the decision is its argmax. `length_normalize` divides by token count, which
    matters when candidates have different lengths ("billing" vs "account access").
    """
    lm.reset_counters()
    seqs = [tokenize(l) + [EOS] for l in labels]
    scores = lm.score_sequences(ticket, "label", seqs)
    if length_normalize:
        scores = [s / len(q) for s, q in zip(scores, seqs)]
    m = max(scores)
    exp = [math.exp(s - m) for s in scores]
    z = sum(exp)
    probs = {l: e / z for l, e in zip(labels, exp)}
    best = max(probs, key=probs.get)
    raw = ", ".join(f"{l} -> {p:.2f}" for l, p in sorted(probs.items(), key=lambda kv: -kv[1]))
    return Result(best, raw, lm.forward_passes, lm.positions, probs)
