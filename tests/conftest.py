"""Shared fixtures. No test here needs GROQ_API_KEY or network: the LLM is
replaced by a fake, retrieval runs against the local Chroma index, and
profile data goes to a temp file."""
import pytest
from langchain_core.documents import Document

from src.profile import profile_store
from src.rag.chain import ActivityRecommendation


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    """Keep test runs out of the LangSmith project."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


@pytest.fixture
def user_data(tmp_path, monkeypatch):
    path = tmp_path / "user_data.json"
    monkeypatch.setattr(profile_store, "DATA_PATH", str(path))
    return path


def make_doc(title: str, category: str = "connect", minutes: int = 40) -> Document:
    return Document(
        page_content=f"{title}. A test activity.",
        metadata={
            "id": title.lower().replace(" ", "_"),
            "title": title,
            "category": category,
            "duration_minutes": minutes,
            "energy_level": "medium",
            "location": "home",
            "required_materials": "paper and pen",
        },
    )


def make_recommendation(title: str, category: str = "connect") -> ActivityRecommendation:
    return ActivityRecommendation(
        title=title,
        reason="It fits.",
        duration_minutes=40,
        category=category,
        why_it_matches=["matches interests"],
        instructions=["Step one", "Step two"],
        materials=["paper"],
        phone_free_message="Put the phone away and begin.",
    )


class FakeLLM:
    """Stands in for the structured-output ChatGroq runnable."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.result
