"""Evaluation script for the distillation pipeline.

Reports, on the held-out test split saved by train_student.py:
- accuracy, macro precision/recall/F1, and a confusion matrix for the
  student's category classifier
- a measured (not fabricated) latency comparison between one student
  prediction and one teacher (Groq) API call, run on this machine, on this
  network, right now - if GROQ_API_KEY isn't set the teacher side is
  reported as "not measured" rather than guessed.

Run: python -m src.evaluation.evaluate
"""
import json
import os
import time

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import STUDENT_MODEL_DIR, load_settings
from src.distillation.student import predict as student_predict


def load_split():
    path = os.path.join(STUDENT_MODEL_DIR, "split.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run train_student.py first."
        )
    return joblib.load(path)


def evaluate_category_classifier() -> dict:
    data = load_split()
    records, idx_test = data["records"], data["idx_test"]
    category_pipeline = joblib.load(os.path.join(STUDENT_MODEL_DIR, "category_pipeline.joblib"))

    X_test = [records[i]["query"] for i in idx_test]
    y_true = [records[i]["category"] for i in idx_test]
    y_pred = category_pipeline.predict(X_test)

    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def measure_latency(n: int = 5) -> dict:
    """Time n student predictions on a fixed sample query. Times one
    teacher call too, only if a Groq key is configured - we do not
    estimate/fabricate a number when there's no key to actually call."""
    sample_query = "I have 45 minutes and want to do something useful with my grandson."

    # warm the cached pipeline load before timing
    student_predict(sample_query)
    start = time.perf_counter()
    for _ in range(n):
        student_predict(sample_query)
    student_latency_ms = (time.perf_counter() - start) / n * 1000

    settings = load_settings()
    teacher_latency_ms = None
    if settings.groq_api_key:
        from src.distillation.teacher import get_teacher_llm, label_query

        llm = get_teacher_llm(settings)
        start = time.perf_counter()
        label_query(sample_query, llm=llm)
        teacher_latency_ms = (time.perf_counter() - start) * 1000

    return {
        "student_latency_ms": round(student_latency_ms, 2),
        "teacher_latency_ms": round(teacher_latency_ms, 2) if teacher_latency_ms else None,
        "teacher_measured": teacher_latency_ms is not None,
    }


def print_confusion_matrix(labels: list[str], cm: list[list[int]]) -> None:
    width = max(len(l) for l in labels) + 2
    header = " " * width + "".join(f"{l[:6]:>8}" for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        print(f"{label:<{width}}" + "".join(f"{v:>8}" for v in row))


if __name__ == "__main__":
    print("=== Student category classifier (held-out test set) ===")
    report = evaluate_category_classifier()
    print(f"n_test: {report['n_test']}")
    print(f"accuracy: {report['accuracy']:.3f}")
    print(f"macro precision: {report['macro_precision']:.3f}")
    print(f"macro recall: {report['macro_recall']:.3f}")
    print(f"macro F1: {report['macro_f1']:.3f}")
    print("\nconfusion matrix (rows=true, cols=predicted):")
    print_confusion_matrix(report["labels"], report["confusion_matrix"])

    print("\n=== Latency: student vs teacher (measured on this machine) ===")
    latency = measure_latency()
    print(f"student (avg of 5 calls): {latency['student_latency_ms']} ms")
    if latency["teacher_measured"]:
        print(f"teacher (1 Groq API call): {latency['teacher_latency_ms']} ms")
        speedup = latency["teacher_latency_ms"] / latency["student_latency_ms"]
        print(f"student is ~{speedup:.0f}x faster than one teacher call")
    else:
        print("teacher: not measured (GROQ_API_KEY not set)")
