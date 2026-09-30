"""Runtime student classifier: loads the trained pipelines and predicts
category/intent for a user query with no network call.

This is what app.py actually calls on every request - the teacher (Groq
API) is never invoked live, only during the offline data-generation phase.
"""
import os
from dataclasses import dataclass
from functools import lru_cache

import joblib
from langsmith import traceable

from src.config import STUDENT_MODEL_DIR


class StudentNotTrainedError(Exception):
    pass


@dataclass
class StudentPrediction:
    category: str
    intent: str


@lru_cache(maxsize=1)
def _load_pipelines():
    category_path = os.path.join(STUDENT_MODEL_DIR, "category_pipeline.joblib")
    intent_path = os.path.join(STUDENT_MODEL_DIR, "intent_pipeline.joblib")
    if not (os.path.exists(category_path) and os.path.exists(intent_path)):
        raise StudentNotTrainedError(
            "Student model not found. Run "
            "`python -m src.distillation.generate_training_data` then "
            "`python -m src.distillation.train_student` first."
        )
    return joblib.load(category_path), joblib.load(intent_path)


@traceable(name="student_classifier", run_type="tool")
def predict(query: str) -> StudentPrediction:
    category_pipeline, intent_pipeline = _load_pipelines()
    category = category_pipeline.predict([query])[0]
    intent = intent_pipeline.predict([query])[0]
    return StudentPrediction(category=category, intent=intent)


if __name__ == "__main__":
    for q in [
        "Give me something I can do with my grandson for 30 minutes.",
        "I have one hour and want to do something useful at home.",
        "I feel tired today and don't want to go outside.",
    ]:
        pred = predict(q)
        print(f"{q!r} -> category={pred.category}, intent={pred.intent}")
