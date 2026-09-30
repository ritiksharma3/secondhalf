"""RAG chain + pipeline tests. The LLM is replaced by FakeLLM, so these run
without GROQ_API_KEY and are deterministic."""
import pytest

from src.config import Settings, require_groq_key
from src.rag import chain
from src.rag.chain import InvalidRecommendationError, build_recommendation, check_grounding
from src.recommendation import recommender

from tests.conftest import FakeLLM, make_doc, make_recommendation

RETRIEVED = [(make_doc("Teach a mathematics concept"), 0.8), (make_doc("Tend to the garden", "physical_outdoor"), 0.6)]


def test_prompt_contains_retrieved_context_and_user_situation():
    llm = FakeLLM(make_recommendation("Teach a mathematics concept"))
    build_recommendation(RETRIEVED, 45, "medium", "something useful", ["Morning walk"], llm=llm)
    prompt = "\n".join(m.content for m in llm.calls[0])
    assert "Teach a mathematics concept" in prompt
    assert "45 minutes" in prompt and "Morning walk" in prompt
    assert "source of truth" in prompt


def test_non_structured_llm_output_is_rejected():
    with pytest.raises(InvalidRecommendationError):
        build_recommendation(RETRIEVED, 45, "medium", "", [], llm=FakeLLM(None))


def test_empty_retrieval_is_rejected():
    with pytest.raises(ValueError):
        build_recommendation([], 45, "medium", "", [], llm=FakeLLM(None))


def test_light_personalization_is_grounded():
    """Spec section 21: a reasonable personalization of a retrieved title passes."""
    ok, match = check_grounding(make_recommendation("Teach a mathematics concept to your grandson"), RETRIEVED)
    assert ok and match == "Teach a mathematics concept"


def test_unsupported_activity_is_flagged():
    """Spec section 21: gardening + maths retrieved, 'pottery class' suggested -> unsupported."""
    ok, _ = check_grounding(make_recommendation("Join a pottery class"), RETRIEVED)
    assert not ok


def test_missing_groq_key_gives_clear_error():
    settings = Settings(None, "m", None, False, "p")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        require_groq_key(settings)


def test_full_pipeline_with_fake_llm(monkeypatch):
    """student -> retrieval -> (fake) LLM -> grounding, end to end."""
    def fake_llm_factory():
        return FakeLLMPicksFirst()

    class FakeLLMPicksFirst:
        def invoke(self, messages):
            prompt = messages[-1].content
            first_title = prompt.split("1. Title: ")[1].split("\n")[0]
            return make_recommendation(first_title)

    monkeypatch.setattr(chain, "get_recommendation_llm", fake_llm_factory)
    result = recommender.get_recommendation(
        free_text="I want to learn something new.",
        duration_minutes=30,
        energy_level="medium",
        avoid_titles=["Tend to the garden"],
    )
    assert result.student_prediction.category == "learn"
    assert result.category_used == "learn"
    assert result.retrieved and result.is_grounded
    assert "Tend to the garden" not in [d.metadata["title"] for d, _ in result.retrieved]


def test_explicit_category_overrides_student(monkeypatch):
    monkeypatch.setattr(chain, "get_recommendation_llm", lambda: FakeLLM(make_recommendation("x")))
    result = recommender.get_recommendation(
        free_text="I want to learn something new.", duration_minutes=30, energy_level="medium",
        category_override="physical_outdoor",
    )
    assert result.student_prediction.category == "learn"
    assert result.category_used == "physical_outdoor"
    assert {d.metadata["category"] for d, _ in result.retrieved} == {"physical_outdoor"}
