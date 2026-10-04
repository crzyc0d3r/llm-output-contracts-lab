"""Synthetic support-ticket routing data.

Three teams, each with its own vocabulary, plus a share of "mixed" tickets that
mention two teams' topics so the model has genuinely uncertain cases. Generation
is seeded, so every run sees the same corpus.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

LABELS = ("billing", "technical support", "account access")

_TOPICS = {
    "billing": [
        "i was charged twice for the same subscription",
        "please refund my last payment",
        "my invoice shows the wrong price",
        "the card payment went through two times",
        "why did my subscription renew at a higher price",
        "i need a copy of my invoice for the refund",
        "you billed my card after i cancelled",
        "the charge on my statement is wrong",
    ],
    "technical support": [
        "the app crashes when i open the dashboard",
        "i get an error when uploading a file",
        "the page is very slow and will not load",
        "sync is broken after the latest update",
        "the install fails with an error code",
        "the export button does nothing",
        "notifications stopped working on my phone",
        "the app freezes and shows a blank screen",
    ],
    "account access": [
        "i cannot log in to my account",
        "my password reset email never arrives",
        "my account is locked after too many attempts",
        "the two factor code is not accepted",
        "i lost access to the email on my account",
        "sign in keeps saying my password is wrong",
        "i need to change the owner of my login",
        "the login link has expired",
    ],
}

_OPENERS = ["", "hi ,", "hello ,", "urgent :", "help ,", "quick question :"]
_CLOSERS = ["", "thanks", "please help", "this is urgent", "thank you"]

# Response templates per output style. `{label}` is replaced by the team name.
PROSE_TEMPLATES = [
    "based on the customer's query , this ticket should go to {label} .",
    "this looks like a {label} issue .",
    "i would route this ticket to {label} .",
]
JSON_TEMPLATE = '{{ " team " : " {label} " }}'
LABEL_TEMPLATE = "{label}"


@dataclass(frozen=True)
class Ticket:
    text: str
    label: str            # the team that should own it
    mixed_with: str | None = None


def _compose(rng: random.Random, label: str, mixed_with: str | None) -> str:
    parts = [rng.choice(_OPENERS), rng.choice(_TOPICS[label])]
    if mixed_with:
        parts += ["and", rng.choice(_TOPICS[mixed_with])]
    parts.append(rng.choice(_CLOSERS))
    return " ".join(p for p in parts if p)


def make_tickets(n: int, seed: int, mixed_rate: float = 0.15) -> list[Ticket]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        label = rng.choice(LABELS)
        mixed = rng.choice([l for l in LABELS if l != label]) if rng.random() < mixed_rate else None
        out.append(Ticket(_compose(rng, label, mixed), label, mixed))
    return out


def training_pairs(tickets: list[Ticket], seed: int = 0) -> list[tuple[str, str, str]]:
    """(ticket text, style, response text) for every ticket in every style."""
    rng = random.Random(seed)
    pairs = []
    for t in tickets:
        pairs.append((t.text, "prose", rng.choice(PROSE_TEMPLATES).format(label=t.label)))
        pairs.append((t.text, "json", JSON_TEMPLATE.format(label=t.label)))
        pairs.append((t.text, "label", LABEL_TEMPLATE.format(label=t.label)))
    return pairs


def train_test(seed: int = 7) -> tuple[list[Ticket], list[Ticket]]:
    return make_tickets(600, seed), make_tickets(150, seed + 1000, mixed_rate=0.3)
