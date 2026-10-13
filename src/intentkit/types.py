from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

# intent name -> example utterances
Intents = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class Prediction:
    """Result for one utterance. `intent` is None when the input is out of scope."""

    text: str
    intent: str | None
    confidence: float
    best_guess: str
    scores: dict[str, float] = field(repr=False)

    @property
    def out_of_scope(self) -> bool:
        return self.intent is None


def validate_intents(intents: Intents) -> list[str]:
    if len(intents) < 2:
        raise ValueError("need at least two intents")
    for name, examples in intents.items():
        if isinstance(examples, str) or len(examples) == 0:
            raise ValueError(f"intent {name!r} needs a non-empty list of example strings")
    return sorted(intents)
