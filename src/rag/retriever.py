"""Builds a retrieval query from user context and runs similarity search
against the Chroma activity collection.

This is the "DATASET -> DOCUMENT CREATION -> EMBEDDINGS -> CHROMA ->
RETRIEVER -> TOP-K DOCUMENTS" half of the RAG pipeline (see spec section 7).
We deliberately build a descriptive natural-language query instead of
passing the raw user click-path straight into similarity search, because a
richer query embeds closer to the activities that actually match.
"""
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langsmith import traceable

from src.config import CATEGORY_BOOST, CHROMA_DIR, RETRIEVAL_K
from src.embeddings.embedding_service import get_embedding_function
from src.rag.ingestion import COLLECTION_NAME


@dataclass
class RetrievalContext:
    """Everything needed to build a retrieval query. `category`/`intent`
    normally come from the student classifier (Phase 4); duration/energy
    come directly from the UI; interests/avoid_activities come from the
    user profile and recent history."""

    category: str | None = None
    duration_minutes: int | None = None
    energy_level: str | None = None
    social_preference: str | None = None
    interests: list[str] = field(default_factory=list)
    avoid_titles: list[str] = field(default_factory=list)
    free_text: str = ""
    # True when the user explicitly picked the category (a UI button): only
    # that category is searched. False when it's the student's guess: the
    # category just boosts ranking, so a strong match elsewhere can still win.
    category_is_hard: bool = True


def build_query(ctx: RetrievalContext) -> str:
    """Turn a RetrievalContext into one natural-language sentence.

    Deliberately soft/semantic only (interests, social mood, free text) -
    structural facts like duration/energy/category read as keyword-stuffed
    filter jargon to a small embedding model and measurably hurt retrieval
    (verified: a clause-stuffed query scored 0.35 cosine against the ideal
    "teach mathematics" activity vs. 0.60 for a plain sentence). Those hard
    constraints are applied separately as a metadata filter in retrieve().
    """
    parts = []
    if ctx.interests:
        parts.append(f"enjoys {', '.join(ctx.interests)}")
    if ctx.social_preference:
        parts.append(f"wants {ctx.social_preference} social interaction")
    if ctx.free_text:
        parts.append(ctx.free_text)

    if not parts:
        return "a meaningful offline activity for a retired adult"
    return "An activity for a retired adult who " + " and ".join(parts) + "."


def build_metadata_filter(ctx: RetrievalContext, include_category: bool = True) -> dict | None:
    """Build a Chroma `where` filter for the hard constraints that
    similarity search handles poorly: category (exact match) and duration
    (a tolerance window, since a "45 minute" request should still surface a
    40-minute activity)."""
    clauses = []
    if ctx.category and include_category:
        clauses.append({"category": {"$eq": ctx.category}})
    if ctx.duration_minutes:
        clauses.append({"duration_minutes": {"$gte": max(5, ctx.duration_minutes - 20)}})
        clauses.append({"duration_minutes": {"$lte": ctx.duration_minutes + 20}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def get_vectorstore(persist_dir: str = CHROMA_DIR) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


@traceable(name="rag_retrieval", run_type="retriever")
def retrieve(
    ctx: RetrievalContext, k: int = RETRIEVAL_K, vectorstore: Chroma | None = None
) -> tuple[str, list[tuple[Document, float]]]:
    """Build the query, run similarity search under the metadata filter,
    and return (query, [(document, similarity_score), ...]) so callers
    (and the technical-debug UI) can show both the query and the scored
    results.

    similarity_search_with_score returns cosine *distance* on this
    collection (hnsw:space="cosine"); we convert to similarity
    (1 - distance) so higher is always better for callers, matching the
    printed/UI convention.

    Falls back to a looser filter (category only, then no filter) if the
    strict filter returns nothing - a real request should never come back
    empty just because duration didn't line up exactly.

    When the category is only the student's guess (category_is_hard=False)
    it becomes a ranking boost (CATEGORY_BOOST) instead of a filter.
    """
    store = vectorstore or get_vectorstore()
    query = build_query(ctx)
    avoid = {t.lower() for t in ctx.avoid_titles}

    def search(where: dict | None) -> list[tuple[Document, float]]:
        raw = store.similarity_search_with_score(query, k=k + len(avoid), filter=where)
        return [(doc, 1 - dist) for doc, dist in raw if doc.metadata["title"].lower() not in avoid]

    if ctx.category and not ctx.category_is_hard:
        # Soft mode: pool the student's category with every other category
        # (same duration window), then rank with a small boost for the
        # student's category. Scores returned stay the raw similarities.
        pool = {doc.metadata["id"]: (doc, score)
                for where in (build_metadata_filter(ctx), build_metadata_filter(ctx, include_category=False))
                for doc, score in search(where)}
        ranked = sorted(
            pool.values(),
            key=lambda pair: pair[1] + (CATEGORY_BOOST if pair[0].metadata["category"] == ctx.category else 0),
            reverse=True,
        )
        if ranked:
            return query, ranked[:k]
        fallbacks = (None,)
    else:
        fallbacks = (
            build_metadata_filter(ctx),
            {"category": {"$eq": ctx.category}} if ctx.category else None,
            None,
        )

    for where in fallbacks:
        results = search(where)
        if results:
            return query, results[:k]

    return query, []


if __name__ == "__main__":
    demo_ctx = RetrievalContext(
        category="connect",
        duration_minutes=45,
        energy_level="medium",
        social_preference="high",
        interests=["teaching", "mathematics"],
        avoid_titles=["gardening"],
    )
    q, docs = retrieve(demo_ctx)
    print(f"Query: {q}\n")
    for doc, score in docs:
        print(f"{score:.3f}  {doc.metadata['title']}  ({doc.metadata['category']}, "
              f"{doc.metadata['duration_minutes']} min)")
