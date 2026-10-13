import json
from pathlib import Path

import pytest

from intentkit import GENERIC_NEGATIVES, IntentDetector, KNNIntentDetector, Prediction, score

INTENTS = {
    "greet": ["hello", "hi there", "good morning", "hey"],
    "bye": ["goodbye", "see you later", "bye bye", "talk to you soon"],
    "weather": ["what's the weather", "is it going to rain", "how hot is it outside", "weather forecast"],
}


def test_validate_rejects_bad_input():
    with pytest.raises(ValueError):
        KNNIntentDetector().fit({"only_one": ["x"]})
    with pytest.raises(ValueError):
        KNNIntentDetector().fit({"a": "not a list", "b": ["x"]})


def test_knn_predict_and_calibrate():
    det = KNNIntentDetector(threshold=0.0).fit(INTENTS)
    p = det.predict("hallo, guten Morgen")
    assert isinstance(p, Prediction)
    assert p.intent == "greet"
    assert set(p.scores) == set(INTENTS)
    assert [q.intent for q in det.predict(["bye now", "will it snow tomorrow?"])] == ["bye", "weather"]

    dev = [("en", "hey you", "greet"), ("en", "farewell", "bye"), ("en", "what is 2+2", None)]
    det.calibrate(dev)
    assert det.predict("what is 2+2").out_of_scope


@pytest.fixture(scope="module")
def setfit_det():
    return IntentDetector(num_iterations=2).fit(INTENTS, negatives=GENERIC_NEGATIVES[:8])


def test_setfit_fit_predict(setfit_det):
    preds = setfit_det.predict(["hi!", "see ya", "is it raining?"])
    assert [p.best_guess for p in preds] == ["greet", "bye", "weather"]
    assert setfit_det.predict([]) == []
    # the out-of-scope class never leaks out as an intent
    assert all(p.intent in (*INTENTS, None) for p in setfit_det.predict(GENERIC_NEGATIVES))


def test_setfit_save_load(setfit_det, tmp_path: Path):
    setfit_det.save(tmp_path / "m")
    assert json.loads((tmp_path / "m" / "intentkit.json").read_text())["labels"] == sorted(INTENTS)
    loaded = IntentDetector.load(tmp_path / "m")
    texts = ["hello friend", "will it be sunny"]
    assert [p.intent for p in loaded.predict(texts)] == [p.intent for p in setfit_det.predict(texts)]


def test_score():
    examples = [("en", "a", "x"), ("en", "b", None), ("de", "c", "y")]
    preds = [
        Prediction("a", "x", 0.9, "x", {}),
        Prediction("b", "x", 0.4, "x", {}),
        Prediction("c", None, 0.3, "y", {}),
    ]
    s = score(examples, preds)
    assert s["all"]["in_scope_acc"] == 0.5
    assert s["all"]["closed_acc"] == 1.0
    assert s["all"]["oos_recall"] == 0.0
    assert s["en"]["n"] == 2
