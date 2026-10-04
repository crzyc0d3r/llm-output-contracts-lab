"""A tiny, transparent conditional language model.

It exists so the three output contracts in `strategies.py` can be compared on the
*same* next-token distribution without a GPU. It is not a transformer, but it has
the interface a decoder-only LLM exposes:

    next_logprobs(ticket, style, prefix)   -> log P(next token | prompt, prefix)
    score_sequences(ticket, style, seqs)   -> log P(seq | prompt) for each seq, teacher-forced

How it predicts the next response token:

    logit(w) = log P_ngram(w | style, previous two response tokens) # fluency / format
             + lam * mean_{c in ticket words} PMI(c, w)             # conditioning on the ticket

The n-gram part (interpolated trigram/bigram/unigram) learns the response
templates; the PMI part learns which ticket words go with which output tokens,
which is how "charged" and "refund" end up pushing probability toward "billing".

It also counts the work done, the way you would budget an LLM call:
`forward_passes` (sequential model invocations) and `positions` (tokens pushed
through the model).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

EOS = "<eos>"
BOS = "<s>"
STYLE_MARKERS = {"prose": "<prose>", "json": "<json>", "label": "<label>"}


def tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class TinyLM:
    lam: float = 1.0                 # weight of the ticket-conditioning term
    k: float = 0.01                  # add-k smoothing for the n-gram tables
    l3: float = 0.80
    l2: float = 0.15
    pmi_clip: float = 4.0
    forward_passes: int = 0
    positions: int = 0
    _fed: int = 0                    # response tokens already pushed through the model

    vocab: list[str] = field(default_factory=list)
    _idx: dict[str, int] = field(default_factory=dict)
    _tri: dict = field(default_factory=dict)
    _bi: dict = field(default_factory=dict)
    _uni: np.ndarray | None = None
    _pmi: dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    def fit(self, pairs: list[tuple[str, str, str]]) -> "TinyLM":
        """pairs: (ticket text, style, response text)."""
        responses = [(tokenize(t), s, tokenize(r) + [EOS]) for t, s, r in pairs]
        self.vocab = sorted({w for _, _, r in responses for w in r})
        self._idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)

        tri, bi = defaultdict(lambda: np.zeros(V)), defaultdict(lambda: np.zeros(V))
        uni = np.zeros(V)
        co = defaultdict(lambda: np.zeros(V))
        word_count: Counter = Counter()
        for ticket, style, resp in responses:
            ctx = [BOS, STYLE_MARKERS[style]]
            for w in resp:
                j = self._idx[w]
                tri[(style, ctx[-2], ctx[-1])][j] += 1
                bi[(style, ctx[-1])][j] += 1
                uni[j] += 1
                ctx.append(w)
            for c in set(ticket):
                word_count[c] += 1
                for w in resp:
                    co[c][self._idx[w]] += 1

        self._tri = {k: self._normalize(v) for k, v in tri.items()}
        self._bi = {k: self._normalize(v) for k, v in bi.items()}
        self._uni = self._normalize(uni)
        p_w = self._uni
        self._pmi = {}
        for c, row in co.items():
            if word_count[c] < 2:
                continue
            p_w_given_c = self._normalize(row)
            self._pmi[c] = np.clip(np.log(p_w_given_c / p_w), -self.pmi_clip, self.pmi_clip)
        return self

    def _normalize(self, counts: np.ndarray) -> np.ndarray:
        c = counts + self.k
        return c / c.sum()

    # ----------------------------------------------------------------- inference
    def _logits(self, ticket_words: list[str], style: str, ctx: list[str]) -> np.ndarray:
        p2, p1 = ctx[-2], ctx[-1]
        bi = self._bi.get((style, p1), self._uni)
        p = (1 - self.l3 - self.l2) * self._uni
        p = p + self.l2 * bi
        p = p + self.l3 * self._tri.get((style, p2, p1), bi)
        logit = np.log(p)
        cond = [self._pmi[c] for c in ticket_words if c in self._pmi]
        if cond:
            logit = logit + self.lam * np.mean(cond, axis=0)
        return logit

    @staticmethod
    def _log_softmax(x: np.ndarray) -> np.ndarray:
        m = x.max()
        return x - (m + math.log(np.exp(x - m).sum()))

    def next_logprobs(self, ticket: str, style: str, prefix: list[str]) -> np.ndarray:
        """One decoding step: a single forward pass that adds one new position."""
        if self.forward_passes == 0:                 # first call prefills the prompt
            self.positions += len(tokenize(ticket)) + 1
        # every token appended since the last call must be pushed through the model,
        # including tokens a constrained decoder fast-forwarded without sampling
        self.positions += len(prefix) - self._fed
        self._fed = len(prefix)
        self.forward_passes += 1
        ctx = [BOS, STYLE_MARKERS[style], *prefix]
        return self._log_softmax(self._logits(tokenize(ticket), style, ctx))

    def score_sequences(self, ticket: str, style: str, seqs: list[list[str]]) -> list[float]:
        """Teacher-forced log-likelihood of each candidate continuation.

        A transformer scores all candidates in ONE batched forward pass (prompt
        plus every candidate's tokens, no sampling loop), so that is what is counted.
        """
        self.forward_passes += 1
        self.positions += len(tokenize(ticket)) + 1 + sum(len(s) for s in seqs)
        words = tokenize(ticket)
        out = []
        for seq in seqs:
            ctx = [BOS, STYLE_MARKERS[style]]
            total = 0.0
            for w in seq:
                lp = self._log_softmax(self._logits(words, style, ctx))
                total += float(lp[self._idx[w]])
                ctx.append(w)
            out.append(total)
        return out

    def token_id(self, w: str) -> int:
        return self._idx[w]

    def reset_counters(self) -> None:
        self.forward_passes = 0
        self.positions = 0
        self._fed = 0
