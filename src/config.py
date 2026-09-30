"""Centralized environment/config loading for SecondHalf.

All settings load from .env here and nowhere else, so there is exactly one
place to change LLM provider, model name, or tracing behavior.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
STUDENT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "student_classifier")

ACTIVITIES_CSV = os.path.join(DATA_DIR, "activities.csv")
TEACHER_DATA_JSONL = os.path.join(DATA_DIR, "teacher_training_data.jsonl")
EVALUATION_QUESTIONS_JSON = os.path.join(DATA_DIR, "evaluation_questions.json")
USER_DATA_JSON = os.path.join(DATA_DIR, "user_data.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 5
# Similarity bonus for activities in the student-predicted category when the
# user didn't pick a category themselves (soft category prior, not a filter).
CATEGORY_BOOST = 0.05


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    langchain_api_key: str | None
    langchain_tracing_enabled: bool
    langchain_project: str


def load_settings() -> Settings:
    """Load settings from environment. Does NOT require any key to be set -
    callers that actually need GROQ_API_KEY (e.g. the teacher/recommender)
    validate its presence themselves at call time, not here."""
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY") or None,
        langchain_tracing_enabled=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "secondhalf"),
    )


def require_groq_key(settings: Settings) -> str:
    """Raise a clear error only when a phase that actually needs Groq runs
    without a key, instead of failing at import time."""
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://console.groq.com/keys before running this step."
        )
    return settings.groq_api_key
