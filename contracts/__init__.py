"""Three output contracts for fixed-choice LLM decisions: free decoding, grammar-constrained
structured output, and candidate scoring, compared on one tiny conditional language model."""
from .data import LABELS
from .lm import TinyLM
from .strategies import Result, constrained_decode, free_decode, score_candidates

__all__ = ["LABELS", "TinyLM", "Result", "constrained_decode", "free_decode", "score_candidates"]
