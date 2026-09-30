"""Retrieval tests against the real local Chroma index (no API key needed;
the embedding model runs locally)."""
import pytest

from src.config import RETRIEVAL_K
from src.data.data_loader import load_activities
from src.rag.ingestion import build_document
from src.rag.prompts import format_retrieved_context
from src.rag.retriever import RetrievalContext, build_metadata_filter, build_query, retrieve

from tests.conftest import make_doc


def test_dataset_is_meaningful():
    activities = load_activities()
    assert 150 <= len(activities) <= 300
    assert len({a["category"] for a in activities}) == 10
    assert len({a["id"] for a in activities}) == len(activities)


def test_document_carries_semantics_and_metadata():
    doc = build_document(load_activities()[0])
    assert doc.metadata["title"] in doc.page_content
    assert {"category", "duration_minutes", "energy_level"} <= doc.metadata.keys()


def test_query_is_natural_language():
    ctx = RetrievalContext(interests=["teaching"], social_preference="high", free_text="with a friend")
    assert build_query(ctx) == (
        "An activity for a retired adult who enjoys teaching and wants high social interaction and with a friend."
    )
    assert build_query(RetrievalContext()) == "a meaningful offline activity for a retired adult"


def test_metadata_filter_windows_duration():
    f = build_metadata_filter(RetrievalContext(category="learn", duration_minutes=45))
    assert f == {"$and": [
        {"category": {"$eq": "learn"}},
        {"duration_minutes": {"$gte": 25}},
        {"duration_minutes": {"$lte": 65}},
    ]}
    assert build_metadata_filter(RetrievalContext(category="learn"), include_category=False) is None


def test_hard_category_only_returns_that_category():
    _, results = retrieve(RetrievalContext(category="memory", duration_minutes=30, free_text="old stories"))
    assert 0 < len(results) <= RETRIEVAL_K
    assert {doc.metadata["category"] for doc, _ in results} == {"memory"}


def test_soft_category_can_surface_other_categories():
    """Option B: the student's guess only boosts - a strong teaching match
    for Mr. Sharma must beat a weak practical_useful one."""
    ctx = RetrievalContext(
        category="practical_useful", category_is_hard=False, duration_minutes=45,
        interests=["mathematics", "teaching"], social_preference="high",
        free_text="I want to do something useful and preferably with another person.",
    )
    _, results = retrieve(ctx)
    categories = {doc.metadata["category"] for doc, _ in results}
    assert categories - {"practical_useful"}, "soft mode should not restrict to one category"


def test_avoided_titles_never_returned():
    ctx = RetrievalContext(category="physical_outdoor", free_text="gardening", interests=["gardening"])
    _, first = retrieve(ctx)
    avoided = first[0][0].metadata["title"]
    ctx.avoid_titles = [avoided]
    _, second = retrieve(ctx)
    assert avoided not in [doc.metadata["title"] for doc, _ in second]


def test_scores_are_similarities():
    _, results = retrieve(RetrievalContext(free_text="read a book"))
    assert all(-1.0 <= score <= 1.0 for _, score in results)


def test_context_formatting_numbers_candidates():
    text = format_retrieved_context([(make_doc("Teach maths"), 0.9), (make_doc("Tell a story"), 0.8)])
    assert "1. Title: Teach maths" in text and "2. Title: Tell a story" in text
    assert "Similarity score: 0.900" in text
