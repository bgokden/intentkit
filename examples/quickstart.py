"""Train on mostly-English examples (+2 per language), then classify queries in five languages.

    uv run python examples/quickstart.py
"""

import json
import warnings
from pathlib import Path

from intentkit import GENERIC_NEGATIVES, MULTILINGUAL_NEGATIVES, IntentDetector

warnings.filterwarnings("ignore")
data = json.loads((Path(__file__).parent / "data" / "support_intents.json").read_text())

# 10 English examples per intent, plus 2 each in de/es/fr/tr
intents = {name: examples + [t for _, t in data["train_multilingual"][name]] for name, examples in data["train"].items()}

detector = IntentDetector().fit(intents, negatives=GENERIC_NEGATIVES + MULTILINGUAL_NEGATIVES)
detector.calibrate([tuple(x) for x in data["dev"]])  # English dev set picks the OOS thresholds
print(f"calibrated: threshold={detector.threshold} min_similarity={detector.min_similarity}\n")

queries = [
    "Where's my stuff? Ordered it 10 days ago",  # en
    "Ich möchte mein Passwort zurücksetzen",  # de
    "¿Puedo cambiar la dirección de entrega?",  # es
    "Je veux annuler ma commande tout de suite",  # fr
    "Ürün bozuk geldi, paramı geri istiyorum",  # tr
    "Kargom nerede?",  # tr
    "What's the best pizza place nearby?",  # out of scope
    "Wie spät ist es in New York?",  # out of scope (de)
]
for p in detector.predict(queries):
    label = p.intent or f"<out of scope> (best guess: {p.best_guess})"
    print(f"{p.confidence:5.2f}  {label:45}  {p.text}")

detector.save("models/support")
reloaded = IntentDetector.load("models/support")
assert reloaded.predict(queries[1]).intent == detector.predict(queries[1]).intent
print("\nsaved to models/support and reloaded OK")
