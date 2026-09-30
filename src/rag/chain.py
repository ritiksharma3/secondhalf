"""The final RAG step: retrieved context -> prompt -> LLM -> Pydantic-
validated ActivityRecommendation, plus a grounding check.

LCEL pipeline: format retrieved docs -> ChatPromptTemplate -> ChatGroq with
structured output (Pydantic schema enforced via tool-calling under the
hood) -> ActivityRecommendation.
"""
from difflib import SequenceMatcher

from langchain_core.documents import Document
from langsmith import traceable
from pydantic import BaseModel, Field, ValidationError

from src.config import Settings, load_settings, require_groq_key
from src.rag.prompts import format_retrieved_context, recommendation_prompt


class InvalidRecommendationError(Exception):
    """The LLM answered, but not with a valid ActivityRecommendation."""


class ActivityRecommendation(BaseModel):
    title: str = Field(description="The activity's title, personalized for this user")
    reason: str = Field(description="One or two sentences on why this fits this person right now")
    duration_minutes: int = Field(description="Expected duration in minutes")
    category: str = Field(description="The activity's category")
    why_it_matches: list[str] = Field(description="2-4 short bullet points on why this matches the user")
    instructions: list[str] = Field(description="Concrete step-by-step instructions")
    materials: list[str] = Field(description="Materials needed, empty list if none")
    phone_free_message: str = Field(description="A short encouraging line to put the phone away and start")


def get_recommendation_llm(settings: Settings | None = None):
    settings = settings or load_settings()
    api_key = require_groq_key(settings)
    from langchain_groq import ChatGroq

    from groq import BadRequestError

    llm = ChatGroq(model=settings.groq_model, api_key=api_key, temperature=0.3, timeout=30, max_retries=2)
    # Groq occasionally rejects a malformed tool call from the model with an
    # HTTP 400 (seen ~1 in 8 calls). The client only retries 429/5xx, so we
    # retry that specific failure once ourselves.
    return llm.with_structured_output(ActivityRecommendation).with_retry(
        retry_if_exception_type=(BadRequestError,), stop_after_attempt=2
    )


@traceable(name="recommendation_chain", run_type="chain")
def build_recommendation(
    retrieved: list[tuple[Document, float]],
    duration_minutes: int,
    energy_level: str,
    free_text: str,
    avoid_titles: list[str],
    llm=None,
) -> ActivityRecommendation:
    """Run the recommendation chain over already-retrieved documents."""
    if not retrieved:
        raise ValueError("No retrieved activities to recommend from - retrieval returned empty.")

    llm = llm or get_recommendation_llm()
    messages = recommendation_prompt.format_messages(
        duration_minutes=duration_minutes,
        energy_level=energy_level,
        free_text=free_text or "(no specific request, open to suggestions)",
        avoid_titles=", ".join(avoid_titles) if avoid_titles else "none",
        retrieved_context=format_retrieved_context(retrieved),
    )
    try:
        result = llm.invoke(messages)
    except ValidationError as e:
        raise InvalidRecommendationError(f"LLM output failed schema validation: {e}") from e
    if not isinstance(result, ActivityRecommendation):
        raise InvalidRecommendationError("LLM returned no structured recommendation.")
    return result


@traceable(name="grounding_check", run_type="tool")
def check_grounding(
    recommendation: ActivityRecommendation, retrieved: list[tuple[Document, float]], threshold: float = 0.5
) -> tuple[bool, str | None]:
    """Verify the recommended title corresponds to one of the retrieved
    activities (spec section 21). Uses fuzzy string similarity since the
    LLM is allowed to lightly personalize the title. Returns
    (is_grounded, matched_title_or_None)."""
    best_match, best_score = None, 0.0
    for doc, _ in retrieved:
        candidate_title = doc.metadata["title"]
        score = SequenceMatcher(None, recommendation.title.lower(), candidate_title.lower()).ratio()
        if score > best_score:
            best_match, best_score = candidate_title, score

    return best_score >= threshold, best_match
