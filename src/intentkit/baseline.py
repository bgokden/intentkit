from __future__ import annotations

from typing import Sequence

import numpy as np

from .detector import DEFAULT_BASE_MODEL
from .evaluate import Example
from .types import Intents, Prediction, validate_intents


class KNNIntentDetector:
    """No-training baseline: embed the examples, label by nearest neighbours.

    An intent's score is the mean cosine similarity of its `k` closest examples.
    Inputs whose best score is below `threshold` are out of scope.
    """

    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, *, k: int = 3, threshold: float = 0.6):
        self.base_model = base_model
        self.k = k
        self.threshold = threshold
        self.labels: list[str] = []
        self._encoder = None
        self._emb: dict[str, np.ndarray] = {}

    def fit(self, intents: Intents) -> "KNNIntentDetector":
        from sentence_transformers import SentenceTransformer

        self.labels = validate_intents(intents)
        self._encoder = SentenceTransformer(self.base_model)
        self._emb = {name: self._embed(list(intents[name])) for name in self.labels}
        return self

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        q = self._embed(list(texts))
        cols = []
        for name in self.labels:
            sims = np.sort(q @ self._emb[name].T, axis=1)[:, ::-1]
            cols.append(sims[:, : self.k].mean(axis=1))
        return np.stack(cols, axis=1)

    def predict(self, texts):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        out = [self._decide(t, row, self.threshold) for t, row in zip(batch, self.scores(batch))]
        return out[0] if single else out

    def calibrate(self, dev: Sequence[Example]) -> "KNNIntentDetector":
        """Pick the similarity ``threshold`` that maximises accuracy on a labelled dev set."""
        texts = [t for _, t, _ in dev]
        gold = [g for *_, g in dev]
        rows = self.scores(texts)
        best = (-1.0, self.threshold)
        for th in np.arange(0.0, 1.0, 0.01):
            acc = np.mean([self._decide(t, r, th).intent == g for t, r, g in zip(texts, rows, gold)])
            if acc > best[0]:
                best = (acc, float(round(th, 2)))
        self.threshold = best[1]
        return self

    def _decide(self, text: str, row: np.ndarray, threshold: float) -> Prediction:
        best = int(row.argmax())
        return Prediction(
            text=text,
            intent=self.labels[best] if row[best] >= threshold else None,
            confidence=float(row[best]),
            best_guess=self.labels[best],
            scores={label: float(s) for label, s in zip(self.labels, row)},
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
