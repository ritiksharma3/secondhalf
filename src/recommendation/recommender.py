"""Orchestrates the full pipeline: student classification -> retrieval ->
RAG chain -> grounded, structured recommendation.

This is the one function app.py (Streamlit) calls. It also returns every
intermediate artifact (student prediction, retrieval query, retrieved
docs+scores, grounding result) so the "Technical Details" debug panel
(spec section 25) can show the pipeline is really happening, not just the
final answer.
"""
from dataclasses import dataclass

from langchain_core.documents import Document
from langsmith import traceable

from src.distillation.student import StudentPrediction, predict as student_predict
from src.rag.chain import ActivityRecommendation, build_recommendation, check_grounding
from src.rag.retriever import RetrievalContext, retrieve


@dataclass
class RecommendationResult:
    student_prediction: StudentPrediction
    category_used: str | None
    retrieval_query: str
    retrieved: list[tuple[Document, float]]
    recommendation: ActivityRecommendation
    is_grounded: bool
    grounded_match: str | None


@traceable(name="secondhalf_pipeline", run_type="chain")
def get_recommendation(
    free_text: str,
    duration_minutes: int,
    energy_level: str,
    social_preference: str | None = None,
    interests: list[str] | None = None,
    avoid_titles: list[str] | None = None,
    category_override: str | None = None,
) -> RecommendationResult:
    """category_override is the category the user explicitly picked in the
    UI (e.g. the "Learn" button). An explicit choice beats the student's
    guess, but the student still runs so its prediction stays visible in
    the technical panel."""
    avoid_titles = avoid_titles or []
    interests = interests or []

    student_prediction = student_predict(free_text) if free_text.strip() else StudentPrediction(
        category=None, intent=None
    )

    ctx = RetrievalContext(
        category=category_override or student_prediction.category,
        category_is_hard=category_override is not None,
        duration_minutes=duration_minutes,
        energy_level=energy_level,
        social_preference=social_preference,
        interests=interests,
        avoid_titles=avoid_titles,
        free_text=free_text,
    )
    query, retrieved = retrieve(ctx)
    if not retrieved:
        raise ValueError("No matching activities were found - try a different time or category.")

    recommendation = build_recommendation(
        retrieved=retrieved,
        duration_minutes=duration_minutes,
        energy_level=energy_level,
        free_text=free_text,
        avoid_titles=avoid_titles,
    )
    is_grounded, matched = check_grounding(recommendation, retrieved)

    return RecommendationResult(
        student_prediction=student_prediction,
        category_used=ctx.category,
        retrieval_query=query,
        retrieved=retrieved,
        recommendation=recommendation,
        is_grounded=is_grounded,
        grounded_match=matched,
    )


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    result = get_recommendation(
        free_text="I want to do something useful and preferably with another person.",
        duration_minutes=45,
        energy_level="medium",
        social_preference="high",
        interests=["teaching", "mathematics", "gardening"],
        avoid_titles=["Tend to the garden"],
    )
    print(f"Student prediction: category={result.student_prediction.category}, "
          f"intent={result.student_prediction.intent}")
    print(f"\nRetrieval query: {result.retrieval_query}")
    print("\nRetrieved:")
    for doc, score in result.retrieved:
        print(f"  {score:.3f}  {doc.metadata['title']}")
    print(f"\nRecommendation: {result.recommendation.title}")
    print(f"Reason: {result.recommendation.reason}")
    print(f"Grounded: {result.is_grounded} (closest retrieved match: {result.grounded_match})")
