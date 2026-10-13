# intentkit

Few-shot, multilingual intent detection with [SetFit](https://github.com/huggingface/setfit).
Give it a handful of example phrases per intent, mostly in English, and it classifies
customer messages in 50+ languages. It can also say "none of these" when a message
doesn't match any intent.

```python
from intentkit import IntentDetector, GENERIC_NEGATIVES, MULTILINGUAL_NEGATIVES

detector = IntentDetector().fit(
    {
        "track_order":    ["Where is my order?", "Has my package shipped?", "Siparişim nerede?", ...],
        "refund_request": ["I want my money back", "Refund my last purchase", "Je souhaite être remboursé", ...],
        "talk_to_human":  ["Let me speak to a real person", "Connect me with support", ...],
    },
    negatives=GENERIC_NEGATIVES + MULTILINGUAL_NEGATIVES,  # teaches "none of the above"
)

detector.predict("Kargom nerede?")
# Prediction(text='Kargom nerede?', intent='track_order', confidence=0.93, best_guess='track_order')

detector.predict("Wie spät ist es in New York?").intent   # -> None (out of scope)
detector.save("models/acme")                              # one small model per customer
```

## How it works

1. **Body:** a multilingual sentence-transformer
   (`paraphrase-multilingual-MiniLM-L12-v2` by default; any Sentence Transformers model works).
2. **SetFit contrastive fine-tuning:** sentences with the same intent are pulled together
   and different intents are pushed apart. Because the encoder is multilingual,
   that separation carries over to languages you gave few or no examples for.
3. **Head:** logistic regression over the fine-tuned embeddings.
4. **Out-of-scope gates:** an input is rejected (`intent=None`) if
   - it lands in the optional *negatives* class, or
   - its top probability is below `threshold`, or
   - its cosine similarity to the closest training example is below `min_similarity`.

   `detector.calibrate(dev_set)` picks both thresholds from a small labelled dev set.

A **`KNNIntentDetector`** with the same interface is included as a no-training baseline:
it embeds the examples and labels each input by its nearest neighbours.

## Results

All numbers come from the scripts in `examples/` running on a laptop CPU/MPS.
The base model is `paraphrase-multilingual-MiniLM-L12-v2` throughout.

### 1. MASSIVE: 60 intents, few-shot, 5 languages

[Amazon MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) is a realistic
assistant-intent dataset with 60 intents, many of them close to each other
(`alarm_set` / `alarm_query` / `alarm_remove`, …). The same 500 parallel test utterances are
used in every language. The metric is closed-set top-1 accuracy (MASSIVE has no out-of-scope queries).

| Shots per intent | Training data | kNN baseline | SetFit | Δ |
|---|---|---|---|---|
| 8  | English only        | 0.530 | 0.576 | +4.6 |
| 8  | English + 2 per language | 0.551 | **0.673** | **+12.2** |
| 16 | English only        | 0.568 | 0.632 | +6.4 |
| 16 | English + 2 per language | 0.574 | **0.694** | **+12.0** |

Per-language tables are in [`examples/results_massive.md`](examples/results_massive.md).

- **SetFit beats the kNN baseline in every setting**, and the gap grows with harder,
  many-intent problems.
- **Two examples per language help SetFit far more than kNN** (+9.7 points vs +2.1 at 8-shot).
  Contrastive training uses them to align the languages; kNN only gets 2 more neighbours.
  German improves most for SetFit: 0.48 → 0.63.

### 2. Customer-support intents with out-of-scope queries

[`examples/data/support_intents.json`](examples/data/support_intents.json) is a small,
hand-written support set: 6 intents and 10 English examples each. The test set is 110 queries
in en/de/es/fr/tr, of which 20 are out-of-scope ("tell me a joke", "do you sell gift cards?").
Thresholds are calibrated on a separate English dev set; SetFit results are the mean of 3 seeds.

| Training data | Method | Overall | In-scope acc | OOS recall |
|---|---|---|---|---|
| English only | kNN baseline | **0.95** | 0.98 | 0.80 |
| English only | SetFit | 0.93 | 0.96 | 0.82 |
| English only | SetFit + negatives | 0.85 | 0.81 | 1.00 |
| English + 2 per language | kNN baseline | **0.98** | 0.98 | 1.00 |
| English + 2 per language | SetFit | 0.93 | 0.99 | 0.67 |
| English + 2 per language | SetFit + multilingual negatives | **0.98** | 0.99 | 0.95 |

Full table with per-language scores: [`examples/results.md`](examples/results.md).

- **When intents are few and clearly different, kNN is already enough.** SetFit only ties it.
  Start with the baseline and use SetFit when intents start to blur together (see MASSIVE).
- **Plain SetFit is bad at rejecting out-of-scope input.** Fine-tuning pulls *everything* toward
  some intent cluster, and with multilingual training only 67% of out-of-scope queries are rejected.
  An explicit negatives class fixes this.
- **The negatives class needs multilingual intents.** If the intents have English examples only,
  the negatives class rejects roughly 20–25% of valid non-English queries. That happens with
  English-only negatives too, so it isn't caused by the language of the negatives. With 2 examples
  per language it reaches 0.99 in-scope accuracy and 0.95 OOS recall.

### Recommended recipe

1. Write 8–16 example phrases per intent, mostly in your main language.
2. Add **2 examples per intent in each language you care about**. An LLM can generate these.
3. Train SetFit with `negatives=GENERIC_NEGATIVES + MULTILINGUAL_NEGATIVES` plus a few
   near-misses from your own domain ("do you sell gift cards?" for a shop).
4. Run `calibrate()` on a small dev set that includes some out-of-scope queries.
5. Measure accuracy **per language**: the average hides weak languages (German was the weakest here).

## Usage

```bash
uv sync
uv run python examples/quickstart.py                # train, predict in 5 languages, save/load
uv run python examples/benchmark.py                 # support intents + OOS (≈4 min)
uv run python examples/benchmark_massive.py --shots 8   # MASSIVE, 60 intents (≈15 min)
uv run pytest
```

### CLI

```bash
# JSON: {intent: [examples]}, or {"train": {...}, "train_multilingual": {intent: [[lang, text]]}, "dev": [[lang, text, intent|null]]}
uv run intentkit train examples/data/support_intents.json models/support
uv run intentkit predict models/support "Kargom nerede?" "Quel temps fait-il à Paris ?"
{"text": "Kargom nerede?", "intent": "track_order", "confidence": 0.93, "best_guess": "track_order"}
{"text": "Quel temps fait-il à Paris ?", "intent": null, "confidence": 0.01, "best_guess": "track_order"}
```

### API

| | |
|---|---|
| `IntentDetector(base_model, threshold, min_similarity, num_iterations=8, ...)` | SetFit classifier |
| `.fit(intents, negatives=None)` | `intents` is `{name: [examples]}`; `negatives` trains a "none of these" class |
| `.calibrate(dev)` | `dev` is `[(lang, text, intent or None), ...]`; picks the OOS thresholds |
| `.predict(text or [texts])` | returns `Prediction(text, intent, confidence, best_guess, scores)` |
| `.save(dir)` / `IntentDetector.load(dir)` | model + config on disk |
| `KNNIntentDetector(base_model, k=3, threshold)` | no-training baseline, same interface |
| `score(examples, predictions)` | per-language accuracy and OOS recall |

## Multi-tenant ("one model per customer")

Each saved `IntentDetector` is self-contained, about 465 MB with the default body. Training one per
customer takes tens of seconds for a typical support taxonomy (see the fit times in `examples/`).
If you have many customers, a cheaper design is to fine-tune one shared body on all customers'
data and store only a small head per customer. That isn't implemented here.

## Limitations

- These are small experiments: the support test set has only 110 queries, so ±0.02 is noise.
  MASSIVE is the more trustworthy comparison.
- One label per message. For multi-intent messages ("refund *and* close my account"), SetFit
  supports `multi_target_strategy="one-vs-rest"`, which isn't wired up here.
- Fit time grows with (#examples)² × `num_iterations` because of pair generation. Lower
  `num_iterations` for large intent sets.

## License

MIT
