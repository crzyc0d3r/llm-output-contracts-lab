"""Compare three output contracts for a fixed-choice decision.

    python cli.py demo                         one ticket through all three contracts
    python cli.py demo --ticket "my login link has expired"
    python cli.py eval                         greedy decoding over 150 held-out tickets
    python cli.py eval --temperature 1.0       same, with sampling
    python cli.py escalate --threshold 0.7     use the scored probabilities to route or escalate
"""
from __future__ import annotations

import argparse

from contracts.evaluate import build_model, escalation_policy, evaluate, format_table
from contracts.strategies import constrained_decode, free_decode, score_candidates

DEFAULT_TICKET = "I was charged twice for the same subscription."


def demo(ticket: str) -> None:
    lm, _ = build_model()
    print(f"ticket: {ticket!r}")
    print("allowed answers: billing | technical support | account access\n")
    shown = [
        ("1) free decoding (prose)", free_decode(lm, ticket, "prose"),
         "generate a sentence, then regex the team out of it"),
        ("2) structured output (grammar-constrained JSON)", constrained_decode(lm, ticket),
         "generate JSON token by token; illegal tokens are masked at every step"),
        ("   ...with fast-forward of forced tokens", constrained_decode(lm, ticket, fast_forward=True),
         "only the step where the schema leaves a real choice calls the model"),
        ("3) candidate scoring", score_candidates(lm, ticket),
         "score the three allowed answers in one pass; nothing is generated"),
    ]
    for title, r, how in shown:
        print(title)
        print(f"   how     : {how}")
        print(f"   output  : {r.raw}")
        print(f"   decision: {r.label}")
        print(f"   cost    : {r.forward_passes} sequential forward pass(es), {r.positions} token positions\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    d = sub.add_parser("demo")
    d.add_argument("--ticket", default=DEFAULT_TICKET)
    e = sub.add_parser("eval")
    e.add_argument("--temperature", type=float, default=0.0)
    e.add_argument("--seed", type=int, default=0)
    x = sub.add_parser("escalate")
    x.add_argument("--threshold", type=float, default=0.7)
    args = p.parse_args()

    if args.cmd in (None, "demo"):
        demo(getattr(args, "ticket", DEFAULT_TICKET))
    elif args.cmd == "eval":
        lm, test = build_model()
        print(f"{len(test)} held-out tickets ({sum(t.mixed_with is not None for t in test)} mix two teams' topics), "
              f"temperature={args.temperature}\n")
        print(format_table(evaluate(lm, test, args.temperature, args.seed)))
        print("\nacc = matches the primary team; valid = the application got a usable decision;\n"
              "passes = sequential model calls; positions = tokens pushed through the model.")
    elif args.cmd == "escalate":
        lm, test = build_model()
        r = escalation_policy(lm, test, args.threshold)
        print(f"threshold {r.threshold:.2f}: auto-routed {r.auto_routed}/{len(test)} tickets at "
              f"{r.auto_accuracy:.0%} accuracy; escalated {r.escalated} to a human, "
              f"{r.escalated_mixed} of which are mixed-topic ({r.total_mixed} mixed in total)")


if __name__ == "__main__":
    main()
