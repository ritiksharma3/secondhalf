"""Trains the lightweight student classifier on teacher-labeled data.

Per the spec's simplification (section 11): the student's job is
category/intent classification from free text. Energy and duration are
supplied directly by UI controls, not inferred - keeping the student to one
well-defined, explainable task (TF-IDF + Logistic Regression) rather than a
harder multi-output problem.

Two separate TF-IDF + LogisticRegression pipelines are trained: one for
`category` (used directly to drive retrieval) and one for `intent` (shown
in the technical-debug UI for explainability). Same train/test split (fixed
random_state) is reused by evaluate.py so reported metrics are reproducible.
"""
import json
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import STUDENT_MODEL_DIR, TEACHER_DATA_JSONL

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_labeled_data(path: str = TEACHER_DATA_JSONL) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) < 20:
        raise ValueError(
            f"Only {len(records)} labeled examples in {path} - run "
            "generate_training_data.py first (needs GROQ_API_KEY)."
        )
    return records


def make_split(records: list[dict]):
    queries = [r["query"] for r in records]
    categories = [r["category"] for r in records]
    intents = [r["intent"] for r in records]

    idx_train, idx_test = train_test_split(
        range(len(records)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=categories
    )
    return idx_train, idx_test, queries, categories, intents


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train() -> dict:
    records = load_labeled_data()
    idx_train, idx_test, queries, categories, intents = make_split(records)

    X_train = [queries[i] for i in idx_train]
    X_test = [queries[i] for i in idx_test]

    category_pipeline = build_pipeline()
    category_pipeline.fit(X_train, [categories[i] for i in idx_train])

    intent_pipeline = build_pipeline()
    intent_pipeline.fit(X_train, [intents[i] for i in idx_train])

    os.makedirs(STUDENT_MODEL_DIR, exist_ok=True)
    joblib.dump(category_pipeline, os.path.join(STUDENT_MODEL_DIR, "category_pipeline.joblib"))
    joblib.dump(intent_pipeline, os.path.join(STUDENT_MODEL_DIR, "intent_pipeline.joblib"))
    # Persist the split + raw records so evaluate.py scores on the exact
    # same held-out set instead of re-deriving it (and risking drift).
    joblib.dump(
        {"records": records, "idx_train": idx_train, "idx_test": idx_test},
        os.path.join(STUDENT_MODEL_DIR, "split.joblib"),
    )

    train_acc = category_pipeline.score(X_train, [categories[i] for i in idx_train])
    test_acc = category_pipeline.score(X_test, [categories[i] for i in idx_test])

    return {
        "n_total": len(records),
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "category_train_accuracy": train_acc,
        "category_test_accuracy": test_acc,
    }


if __name__ == "__main__":
    stats = train()
    print(json.dumps(stats, indent=2))
    print(f"\nSaved student model to {STUDENT_MODEL_DIR}")
