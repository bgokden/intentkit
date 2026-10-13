from __future__ import annotations

import contextlib
import io
import json
import logging
import random
import tempfile
from pathlib import Path
from typing import Sequence, overload

import numpy as np

from .evaluate import Example
from .types import Intents, Prediction, validate_intents

DEFAULT_BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONFIG_FILE = "intentkit.json"
OOS_LABEL = "__out_of_scope__"


class IntentDetector:
    """Few-shot, multilingual intent classifier built on SetFit.

    An input is rejected as out of scope (``intent=None``) when any of these hold:
      * the top class is the trained out-of-scope class (see ``fit(negatives=...)``)
      * the top intent's probability is below ``threshold``
      * cosine similarity to the nearest training example, in the fine-tuned
        embedding space, is below ``min_similarity``
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        *,
        threshold: float = 0.5,
        min_similarity: float = 0.5,
        num_iterations: int = 8,
        num_epochs: int = 1,
        batch_size: int = 16,
        seed: int = 42,
    ):
        self.base_model = base_model
        self.threshold = threshold
        self.min_similarity = min_similarity
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.labels: list[str] = []
        self._classes: list[str] = []  # labels, plus OOS_LABEL when negatives were given
        self._model = None
        self._train_texts: list[str] = []
        self._train_emb: np.ndarray | None = None

    # ---- training -----------------------------------------------------------

    def fit(self, intents: Intents, negatives: Sequence[str] | None = None) -> "IntentDetector":
        """Train on ``{intent: [examples]}``.

        ``negatives`` are off-topic utterances learned as an explicit
        out-of-scope class, e.g. ``intentkit.GENERIC_NEGATIVES``.
        """
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        self.labels = validate_intents(intents)
        self._classes = self.labels + ([OOS_LABEL] if negatives else [])
        groups = {**intents, **({OOS_LABEL: list(negatives)} if negatives else {})}
        texts = [t for c in self._classes for t in groups[c]]
        labels = [i for i, c in enumerate(self._classes) for _ in groups[c]]

        random.seed(self.seed)
        np.random.seed(self.seed)
        with tempfile.TemporaryDirectory() as tmp, _quiet():
            self._model = SetFitModel.from_pretrained(self.base_model, labels=self._classes)
            args = TrainingArguments(
                output_dir=tmp,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                num_iterations=self.num_iterations,
                save_strategy="no",
                logging_strategy="no",
                report_to="none",
                seed=self.seed,
                show_progress_bar=False,
            )
            Trainer(
                model=self._model,
                args=args,
                train_dataset=Dataset.from_dict({"text": texts, "label": labels}),
            ).train()

        self._train_texts = [t for name in self.labels for t in intents[name]]
        self._train_emb = self._embed(self._train_texts)
        return self

    # ---- inference ----------------------------------------------------------

    @overload
    def predict(self, texts: str) -> Prediction: ...
    @overload
    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...

    def predict(self, texts):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return []
        probs, sims = self._raw(batch)
        out = [self._decide(t, p, s, self.threshold, self.min_similarity) for t, p, s in zip(batch, probs, sims)]
        return out[0] if single else out

    def calibrate(self, dev: Sequence[Example]) -> "IntentDetector":
        """Pick ``threshold`` and ``min_similarity`` that maximise accuracy on a
        labelled dev set (gold ``None`` = out of scope). Use data you won't test on."""
        texts = [t for _, t, _ in dev]
        gold = [g for *_, g in dev]
        probs, sims = self._raw(texts)
        best = (-1.0, self.threshold, self.min_similarity)
        for th in np.arange(0.0, 1.0, 0.05):
            for ms in np.arange(0.0, 1.0, 0.05):
                acc = np.mean(
                    [self._decide(t, p, s, th, ms).intent == g for t, p, s, g in zip(texts, probs, sims, gold)]
                )
                # strict ">" keeps the loosest thresholds among ties
                if acc > best[0]:
                    best = (acc, float(round(th, 2)), float(round(ms, 2)))
        _, self.threshold, self.min_similarity = best
        return self

    def nearest_similarity(self, texts: Sequence[str]) -> np.ndarray:
        """Cosine similarity of each text to its closest (in-scope) training example."""
        return (self._embed(list(texts)) @ self._train_emb.T).max(axis=1)

    def _raw(self, batch: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("call fit() or load() first")
        probs = np.asarray(self._model.predict_proba(batch, show_progress_bar=False))
        return probs, self.nearest_similarity(batch)

    def _decide(self, text: str, p: np.ndarray, sim: float, threshold: float, min_sim: float) -> Prediction:
        n = len(self.labels)
        best = int(p[:n].argmax())
        in_scope = int(p.argmax()) < n and p[best] >= threshold and sim >= min_sim
        return Prediction(
            text=text,
            intent=self.labels[best] if in_scope else None,
            confidence=float(p[best]),
            best_guess=self.labels[best],
            scores={label: float(s) for label, s in zip(self.labels, p)},
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.model_body.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )

    # ---- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        if self._model is None:
            raise RuntimeError("nothing to save; call fit() first")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        config = {
            "base_model": self.base_model,
            "threshold": self.threshold,
            "min_similarity": self.min_similarity,
            "labels": self.labels,
            "classes": self._classes,
            "train_texts": self._train_texts,
        }
        (path / CONFIG_FILE).write_text(json.dumps(config, ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "IntentDetector":
        from setfit import SetFitModel

        path = Path(path)
        config = json.loads((path / CONFIG_FILE).read_text())
        det = cls(config["base_model"], threshold=config["threshold"], min_similarity=config["min_similarity"])
        det.labels = config["labels"]
        det._classes = config["classes"]
        with _quiet():
            det._model = SetFitModel.from_pretrained(str(path))
        det._train_texts = config["train_texts"]
        det._train_emb = det._embed(det._train_texts)
        return det


class _quiet:
    """Silence the transformers/setfit training chatter."""

    _names = ("transformers", "setfit", "sentence_transformers", "datasets", "huggingface_hub")

    def __enter__(self):
        # the trainer print()s its loss dicts regardless of log level
        self._stdout = contextlib.redirect_stdout(io.StringIO())
        self._stdout.__enter__()
        self._levels = {n: logging.getLogger(n).level for n in self._names}
        for n in self._names:
            logging.getLogger(n).setLevel(logging.ERROR)
        try:
            from transformers.utils import logging as hf_logging

            self._hf_level = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
        except ImportError:
            self._hf_level = None
        try:
            import datasets

            datasets.disable_progress_bars()
        except ImportError:
            pass

    def __exit__(self, *exc):
        self._stdout.__exit__(*exc)
        for n, level in self._levels.items():
            logging.getLogger(n).setLevel(level)
        if self._hf_level is not None:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity(self._hf_level)
