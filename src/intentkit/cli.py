"""intentkit train <intents.json> <model_dir> | intentkit predict <model_dir> <text>..."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

from .detector import DEFAULT_BASE_MODEL, IntentDetector
from .negatives import GENERIC_NEGATIVES, MULTILINGUAL_NEGATIVES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="intentkit", description="Few-shot multilingual intent detection")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train from a JSON file")
    tr.add_argument("intents", help='JSON: {intent: [examples]}, or {"train": {intent: [examples]}, '
                                    '"train_multilingual": {intent: [[lang, text]]}, "dev": [[lang, text, intent|null]]} '
                                    "(train_multilingual and dev optional)")
    tr.add_argument("model_dir")
    tr.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    tr.add_argument("--no-negatives", action="store_true", help="don't train an out-of-scope class")

    pr = sub.add_parser("predict", help="classify text with a trained model")
    pr.add_argument("model_dir")
    pr.add_argument("text", nargs="*", help="texts to classify (default: one per line on stdin)")

    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    if args.cmd == "train":
        spec = json.load(open(args.intents, encoding="utf-8"))
        intents = {k: list(v) for k, v in spec.get("train", spec).items()}
        for name, pairs in spec.get("train_multilingual", {}).items():
            intents.setdefault(name, []).extend(text for _, text in pairs)
        det = IntentDetector(args.base_model).fit(intents, negatives=None if args.no_negatives else GENERIC_NEGATIVES + MULTILINGUAL_NEGATIVES)
        if "dev" in spec:
            det.calibrate([tuple(x) for x in spec["dev"]])
        det.save(args.model_dir)
        print(f"trained {len(det.labels)} intents -> {args.model_dir} "
              f"(threshold={det.threshold}, min_similarity={det.min_similarity})")
    else:
        det = IntentDetector.load(args.model_dir)
        texts = args.text or [line.strip() for line in sys.stdin if line.strip()]
        for p in det.predict(texts):
            print(json.dumps({"text": p.text, "intent": p.intent, "confidence": round(p.confidence, 4),
                              "best_guess": p.best_guess}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
