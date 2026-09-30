"""Student classifier tests. Uses the trained model in models/ - no API call."""
import pytest

from src.distillation.student import predict
from src.evaluation.evaluate import evaluate_category_classifier

CATEGORIES = {
    "learn", "create", "connect", "contribute", "physical_outdoor",
    "hobbies", "family", "memory", "practical_useful", "mindfulness_reflection",
}


@pytest.mark.parametrize(
    "query, expected",
    [
        ("I want to learn something new.", "learn"),
        ("I want something I can do with my grandson.", "family"),
        ("I want something social.", "connect"),
        ("I have one hour and want to create something.", "create"),
        ("I have 30 minutes and want to do something useful at home.", "practical_useful"),
    ],
)
def test_clear_queries_classified_correctly(query, expected):
    assert predict(query).category == expected


def test_prediction_is_a_known_category():
    pred = predict("anything at all")
    assert pred.category in CATEGORIES
    assert pred.intent


def test_held_out_accuracy_stays_high():
    """Guards against a bad retrain silently shipping."""
    report = evaluate_category_classifier()
    assert report["n_test"] > 0
    assert report["accuracy"] >= 0.85
