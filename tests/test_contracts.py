import json
import random

import numpy as np
import pytest

from contracts.data import LABELS, make_tickets
from contracts.evaluate import build_model, escalation_policy, evaluate
from contracts.lm import tokenize
from contracts.strategies import (constrained_decode, detok_json, free_decode, json_sequences, parse_label,
                                  score_candidates)


@pytest.fixture(scope="module")
def model():
    return build_model()


def test_data_is_deterministic_and_labelled():
    a, b = make_tickets(50, seed=1), make_tickets(50, seed=1)
    assert a == b
    assert {t.label for t in a} <= set(LABELS)
    assert all(t.mixed_with != t.label for t in a)


def test_lm_distribution_is_normalized(model):
    lm, _ = model
    lp = lm.next_logprobs("my password reset email never arrives", "label", [])
    assert np.isclose(np.exp(lp).sum(), 1.0)
    assert lm.vocab[int(np.argmax(lp))] == "account"


def test_lm_counts_passes_and_positions(model):
    lm, _ = model
    lm.reset_counters()
    t = "the app crashes"
    lm.next_logprobs(t, "prose", [])
    lm.next_logprobs(t, "prose", ["based"])
    assert lm.forward_passes == 2
    assert lm.positions == len(tokenize(t)) + 1 + 1


def test_parse_label():
    assert parse_label("this ticket should go to billing .") == "billing"
    assert parse_label("route to technical support .") == "technical support"
    assert parse_label("either billing or account access") is None   # ambiguous
    assert parse_label("i am not sure") is None                       # nothing to parse


def test_detok_json_roundtrip():
    for seq in json_sequences():
        obj = json.loads(detok_json(seq[:-1]))
        assert obj["team"] in LABELS
    assert detok_json(tokenize('{ " team " : " bill')) == '{"team":"bill'


def test_greedy_free_decoding_routes_clean_tickets(model):
    lm, test = model
    clean = [t for t in test if not t.mixed_with][:30]
    for t in clean:
        assert free_decode(lm, t.text, "prose").label == t.label


def test_constrained_decoding_is_always_valid_even_when_sampling(model):
    lm, test = model
    rng = random.Random(0)
    for t in test[:60]:
        r = constrained_decode(lm, t.text, temperature=1.5, rng=rng)
        assert r.label in LABELS
        assert json.loads(r.raw) == {"team": r.label}


def test_free_decoding_can_break_the_contract_when_sampling(model):
    lm, test = model
    rng = random.Random(0)
    results = [free_decode(lm, t.text, "json", temperature=1.5, rng=rng) for t in test[:60]]
    assert any(not r.valid for r in results)


def test_fast_forward_saves_passes_not_answers(model):
    lm, test = model
    for t in test[:40]:
        slow = constrained_decode(lm, t.text)
        fast = constrained_decode(lm, t.text, fast_forward=True)
        assert slow.raw == fast.raw
        assert fast.forward_passes == 1 < slow.forward_passes


def test_scoring_returns_a_distribution_in_one_pass(model):
    lm, _ = model
    r = score_candidates(lm, "I was charged twice for the same subscription.")
    assert r.forward_passes == 1
    assert set(r.probs) == set(LABELS)
    assert sum(r.probs.values()) == pytest.approx(1.0)
    assert r.label == "billing" and r.probs["billing"] > 0.8


def test_scoring_is_less_confident_on_mixed_tickets(model):
    lm, test = model
    conf = lambda ts: np.mean([max(score_candidates(lm, t.text).probs.values()) for t in ts])
    assert conf([t for t in test if t.mixed_with]) < conf([t for t in test if not t.mixed_with]) - 0.15


def test_scoring_only_ever_returns_an_allowed_answer(model):
    lm, test = model
    custom = ("billing", "account access")
    for t in test[:20]:
        assert score_candidates(lm, t.text, labels=custom).label in custom


def test_evaluate_table(model):
    lm, test = model
    rows = {r.name: r for r in evaluate(lm, test)}
    assert rows["candidate scoring"].has_distribution
    assert rows["candidate scoring"].passes == 1
    assert rows["grammar-constrained JSON"].valid_rate == 1.0
    assert all(r.accuracy_clean == 1.0 for r in rows.values())


def test_escalation_threshold_trades_coverage_for_accuracy(model):
    lm, test = model
    lo, hi = escalation_policy(lm, test, 0.5), escalation_policy(lm, test, 0.8)
    assert hi.auto_routed < lo.auto_routed
    assert hi.auto_accuracy >= lo.auto_accuracy
    assert hi.escalated_mixed >= lo.escalated_mixed
