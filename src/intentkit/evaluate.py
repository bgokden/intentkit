from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .types import Prediction

# (language, text, gold intent or None for out-of-scope)
Example = tuple[str, str, str | None]


def score(examples: Sequence[Example], preds: Sequence[Prediction]) -> dict:
    """Per-language metrics.

    in_scope_acc  – share of in-scope queries given the right intent
    oos_recall    – share of out-of-scope queries correctly rejected
    overall_acc   – share of all queries handled correctly
    closed_acc    – top-1 accuracy ignoring the out-of-scope gate (in-scope only)
    """
    buckets: dict[str, list[tuple[str | None, Prediction]]] = defaultdict(list)
    for (lang, _, gold), pred in zip(examples, preds):
        buckets[lang].append((gold, pred))
        buckets["all"].append((gold, pred))
    return {lang: _metrics(rows) for lang, rows in buckets.items()}


def _metrics(rows: Iterable[tuple[str | None, Prediction]]) -> dict:
    rows = list(rows)
    ins = [(g, p) for g, p in rows if g is not None]
    oos = [(g, p) for g, p in rows if g is None]
    return {
        "in_scope_acc": _mean(p.intent == g for g, p in ins),
        "closed_acc": _mean(p.best_guess == g for g, p in ins),
        "oos_recall": _mean(p.intent is None for _, p in oos),
        "overall_acc": _mean(p.intent == g for g, p in rows),
        "n": len(rows),
    }


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
