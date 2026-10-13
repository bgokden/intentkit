"""Compare SetFit against a no-training kNN baseline on multilingual intents.

Every method has its thresholds calibrated on the small English dev set, then is
scored on the held-out test set in five languages (en, de, es, fr, tr).

    uv run python examples/benchmark.py                      # default base model
    uv run python examples/benchmark.py --base-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from intentkit import (
    DEFAULT_BASE_MODEL,
    GENERIC_NEGATIVES,
    MULTILINGUAL_NEGATIVES,
    IntentDetector,
    KNNIntentDetector,
    score,
)

warnings.filterwarnings("ignore")
DATA = Path(__file__).parent / "data" / "support_intents.json"
LANGS = ["en", "de", "es", "fr", "tr"]


def load():
    d = json.loads(DATA.read_text())
    multi = {k: v + [t for _, t in d["train_multilingual"][k]] for k, v in d["train"].items()}
    dev = [tuple(x) for x in d["dev"]]
    test = [tuple(x) for x in d["test"]]
    return {"en-only": d["train"], "en + 2/lang": multi}, dev, test


def run(method: str, intents, dev, test, base_model: str, seed: int):
    t0 = time.time()
    if method == "kNN (no training)":
        det = KNNIntentDetector(base_model).fit(intents)
    elif method == "SetFit":
        det = IntentDetector(base_model, seed=seed).fit(intents)
    elif method == "SetFit + en negatives":
        det = IntentDetector(base_model, seed=seed).fit(intents, negatives=GENERIC_NEGATIVES)
    else:
        negatives = GENERIC_NEGATIVES + MULTILINGUAL_NEGATIVES
        det = IntentDetector(base_model, seed=seed).fit(intents, negatives=negatives)
    fit_s = time.time() - t0
    det.calibrate(dev)
    return score(test, det.predict([t for _, t, _ in test])), fit_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results.md")
    args = ap.parse_args()

    regimes, dev, test = load()
    methods = ["kNN (no training)", "SetFit", "SetFit + en negatives", "SetFit + multilingual negatives"]
    rows = []
    for regime, intents in regimes.items():
        for method in methods:
            seeds = [0] if method.startswith("kNN") else list(range(args.seeds))
            runs = [run(method, intents, dev, test, args.base_model, s) for s in seeds]
            avg = lambda lang, key: float(np.mean([r[0][lang][key] for r in runs]))
            row = {
                "regime": regime,
                "method": method,
                "fit_s": float(np.mean([r[1] for r in runs])),
                **{f"{lang}": avg(lang, "overall_acc") for lang in LANGS + ["all"]},
                "in_scope": avg("all", "in_scope_acc"),
                "closed": avg("all", "closed_acc"),
                "oos": avg("all", "oos_recall"),
            }
            rows.append(row)
            print(f"{regime:12} {method:20} overall={row['all']:.2f} oos={row['oos']:.2f} fit={row['fit_s']:.0f}s", flush=True)

    md = render(rows, args.base_model, args.seeds, len(test))
    with args.out.open("a") as f:
        f.write(md + "\n")
    print("\n" + md)


def render(rows, base_model, seeds, n_test) -> str:
    head = (
        f"### `{base_model}`\n\n"
        f"Test set: {n_test} queries (18 in-scope + 4 out-of-scope per language). "
        f"SetFit numbers are the mean of {seeds} seed(s). Fit time on CPU (Apple M-series).\n\n"
        "| Training data | Method | en | de | es | fr | tr | **Overall** | In-scope acc | Closed-set acc | OOS recall | Fit |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['regime']} | {r['method']} | "
        + " | ".join(f"{r[lang]:.2f}" for lang in LANGS)
        + f" | **{r['all']:.2f}** | {r['in_scope']:.2f} | {r['closed']:.2f} | {r['oos']:.2f} | {r['fit_s']:.0f}s |\n"
        for r in rows
    )
    return head + body


if __name__ == "__main__":
    main()
