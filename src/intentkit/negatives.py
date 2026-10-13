"""Generic, domain-agnostic utterances used to train an explicit out-of-scope class.

SetFit's contrastive fine-tuning pulls every input toward one of the known intent
clusters, so unrelated text often gets a confident (wrong) label. Giving the model
a "none of the above" class built from varied off-topic text fixes most of that.
Add domain-specific near-misses of your own for best results.

Benchmark note: the out-of-scope class only helps once each intent has a couple of
examples per target language. With English-only intents it rejects many valid
non-English queries (see examples/results.md), whichever negatives list is used.
"""

GENERIC_NEGATIVES: list[str] = [
    "hello there",
    "good morning, how are you?",
    "thanks, that's all",
    "ok",
    "what time is it in Tokyo?",
    "who wrote Pride and Prejudice?",
    "translate 'good night' into Italian",
    "what's 17 times 23?",
    "recommend a good movie for tonight",
    "play some relaxing music",
    "how tall is Mount Everest?",
    "write a short poem about the sea",
    "how do I center a div in CSS?",
    "what's the best way to learn piano?",
    "set an alarm for 7am",
    "how many calories are in an apple?",
    "who is the president of Brazil?",
    "suggest a name for my cat",
    "how far is the moon from earth?",
    "what's a good exercise for back pain?",
    "is it healthy to drink coffee every day?",
    "explain quantum computing simply",
]

# Same idea in de/es/fr/tr; combine with GENERIC_NEGATIVES for multilingual intents.
MULTILINGUAL_NEGATIVES: list[str] = [
    "Hallo, wie geht's?",
    "Wer hat die Relativitätstheorie entwickelt?",
    "Hola, ¿qué tal?",
    "¿Cuántos habitantes tiene Madrid?",
    "Bonjour, ça va ?",
    "Quelle est la plus longue rivière du monde ?",
    "Merhaba, nasılsın?",
    "Türkiye'nin en yüksek dağı hangisi?",
]
