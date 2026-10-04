"""Run every output contract over the held-out tickets and compare them."""
from __future__ import annotations

import random
import statistics as st
from dataclasses import dataclass
from typing import Callable

from .data import Ticket, train_test, training_pairs
from .lm import TinyLM
from .strategies import Result, constrained_decode, free_decode, score_candidates


def build_model(seed: int = 7, lam: float = 2.0) -> tuple[TinyLM, list[Ticket]]:
    train, test = train_test(seed)
    return TinyLM(lam=lam).fit(training_pairs(train)), test


@dataclass
class Row:
    name: str
    accuracy: float
    accuracy_clean: float
    accuracy_mixed: float
    valid_rate: float
    passes: float
    positions: float
    has_distribution: bool


def strategies(lm: TinyLM, temperature: float, seed: int) -> dict[str, Callable[[str], Result]]:
    rng = random.Random(seed)
    return {
        "free decode -> prose -> regex": lambda t: free_decode(lm, t, "prose", temperature=temperature, rng=rng),
        "free decode -> JSON -> json.loads": lambda t: free_decode(lm, t, "json", temperature=temperature, rng=rng),
        "grammar-constrained JSON": lambda t: constrained_decode(lm, t, temperature=temperature, rng=rng),
        "constrained JSON + fast-forward": lambda t: constrained_decode(lm, t, fast_forward=True,
                                                                         temperature=temperature, rng=rng),
        "candidate scoring": lambda t: score_candidates(lm, t),
    }


def evaluate(lm: TinyLM, tickets: list[Ticket], temperature: float = 0.0, seed: int = 0) -> list[Row]:
    rows = []
    for name, fn in strategies(lm, temperature, seed).items():
        results = [fn(t.text) for t in tickets]
        ok = [r.label == t.label for r, t in zip(results, tickets)]
        clean = [o for o, t in zip(ok, tickets) if not t.mixed_with]
        mixed = [o for o, t in zip(ok, tickets) if t.mixed_with]
        rows.append(Row(
            name=name,
            accuracy=sum(ok) / len(ok),
            accuracy_clean=sum(clean) / max(1, len(clean)),
            accuracy_mixed=sum(mixed) / max(1, len(mixed)),
            valid_rate=sum(r.valid for r in results) / len(results),
            passes=st.mean(r.forward_passes for r in results),
            positions=st.mean(r.positions for r in results),
            has_distribution=bool(results[0].probs),
        ))
    return rows


@dataclass
class Escalation:
    threshold: float
    auto_routed: int
    auto_accuracy: float
    escalated: int
    escalated_mixed: int
    total_mixed: int


def escalation_policy(lm: TinyLM, tickets: list[Ticket], threshold: float = 0.7) -> Escalation:
    """What a probability buys you: auto-route confident tickets, send the rest to a human."""
    auto, auto_ok, esc, esc_mixed = 0, 0, 0, 0
    for t in tickets:
        r = score_candidates(lm, t.text)
        if r.probs[r.label] >= threshold:
            auto += 1
            auto_ok += r.label == t.label
        else:
            esc += 1
            esc_mixed += t.mixed_with is not None
    return Escalation(threshold, auto, auto_ok / max(1, auto), esc, esc_mixed,
                      sum(t.mixed_with is not None for t in tickets))


def format_table(rows: list[Row]) -> str:
    head = (f"{'contract':<36}{'acc':>6}{'clean':>7}{'mixed':>7}{'valid':>7}"
            f"{'passes':>8}{'positions':>11}  distribution")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.name:<36}{r.accuracy:>6.0%}{r.accuracy_clean:>7.0%}{r.accuracy_mixed:>7.0%}"
                     f"{r.valid_rate:>7.0%}{r.passes:>8.1f}{r.positions:>11.1f}  "
                     f"{'yes' if r.has_distribution else 'no'}")
    return "\n".join(lines)
