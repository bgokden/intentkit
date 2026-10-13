"""Few-shot multilingual intent detection with SetFit."""

from .baseline import KNNIntentDetector
from .detector import DEFAULT_BASE_MODEL, IntentDetector
from .evaluate import Example, score
from .negatives import GENERIC_NEGATIVES, MULTILINGUAL_NEGATIVES
from .types import Intents, Prediction

__all__ = [
    "DEFAULT_BASE_MODEL",
    "Example",
    "GENERIC_NEGATIVES",
    "IntentDetector",
    "Intents",
    "KNNIntentDetector",
    "MULTILINGUAL_NEGATIVES",
    "Prediction",
    "score",
]
