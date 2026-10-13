"""Harder benchmark: Amazon MASSIVE (60 intents, parallel utterances in 50+ languages).

Few-shot setting: K labelled English examples per intent (optionally plus a couple
per target language), then closed-set top-1 accuracy on the same held-out test
utterances in en/de/es/fr/tr. MASSIVE has no out-of-scope queries, so this
measures pure intent discrimination.

    uv run python examples/benchmark_massive.py --shots 8
"""

from __future__ import annotations

import argparse
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

from intentkit import DEFAULT_BASE_MODEL, IntentDetector, KNNIntentDetector

warnings.filterwarnings("ignore")
LANGS = ["en", "de", "es", "fr", "tr"]
DATASET = "mteb/amazon_massive_intent"


def sample_per_intent(split, k: int, rng: random.Random) -> dict[str, list[str]]:
    by_intent = defaultdict(list)
    for row in split:
        by_intent[row["label"]].append(row["text"])
    return {intent: rng.sample(texts, min(k, len(texts))) for intent, texts in sorted(by_intent.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--shots", type=int, default=8, help="English examples per intent")
    ap.add_argument("--extra-shots", type=int, default=2, help="per-language examples in the multilingual regime")
    ap.add_argument("--test-size", type=int, default=500, help="test utterances per language")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results_massive.md")
    args = ap.parse_args()

    data = {lang: load_dataset(DATASET, lang) for lang in LANGS}
    # MASSIVE is parallel: the same id is the same utterance in every language,
    # so every language is tested on identical content
    test_ids = set(random.Random(0).sample(sorted(r["id"] for r in data["en"]["test"]), args.test_size))
    tests = {lang: [(r["text"], r["label"]) for r in data[lang]["test"] if r["id"] in test_ids] for lang in LANGS}

    rows = []
    for regime in ("en-only", f"en + {args.extra_shots}/lang"):
        for method in ("kNN (no training)", "SetFit"):
            per_seed = []
            for seed in range(1 if method.startswith("kNN") else args.seeds):
                rng = random.Random(seed)
                intents = sample_per_intent(data["en"]["train"], args.shots, rng)
                if regime != "en-only":
                    for lang in LANGS[1:]:
                        for intent, texts in sample_per_intent(data[lang]["train"], args.extra_shots, rng).items():
                            intents[intent] = list(intents[intent]) + texts
                t0 = time.time()
                det = (
                    KNNIntentDetector(args.base_model)
                    if method.startswith("kNN")
                    else IntentDetector(args.base_model, seed=seed)
                ).fit(intents)
                fit_s = time.time() - t0
                acc = {}
                for lang, rows_ in tests.items():
                    preds = det.predict([t for t, _ in rows_])
                    acc[lang] = float(np.mean([p.best_guess == g for p, (_, g) in zip(preds, rows_)]))
                acc["all"] = float(np.mean([acc[lang] for lang in LANGS]))
                per_seed.append((acc, fit_s))
            row = {
                "regime": regime,
                "method": method,
                **{k: float(np.mean([a[k] for a, _ in per_seed])) for k in LANGS + ["all"]},
                "fit_s": float(np.mean([f for _, f in per_seed])),
                "n_train": sum(len(v) for v in intents.values()),
            }
            rows.append(row)
            print(f"{regime:12} {method:18} all={row['all']:.3f} fit={row['fit_s']:.0f}s", flush=True)

    md = (
        f"### MASSIVE, {args.shots}-shot, `{args.base_model}`\n\n"
        f"60 intents, {args.test_size} parallel test utterances per language, closed-set top-1 accuracy. "
        f"SetFit numbers are the mean of {args.seeds} seed(s).\n\n"
        "| Training data | Method | Train size | en | de | es | fr | tr | **Mean** | Fit |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
        + "".join(
            f"| {r['regime']} | {r['method']} | {r['n_train']} | "
            + " | ".join(f"{r[lang]:.3f}" for lang in LANGS)
            + f" | **{r['all']:.3f}** | {r['fit_s']:.0f}s |\n"
            for r in rows
        )
    )
    with args.out.open("a") as f:
        f.write(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
