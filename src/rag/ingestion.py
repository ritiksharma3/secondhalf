"""Builds LangChain Documents from activities.csv, embeds them, and
persists them into a Chroma collection on disk.

Pipeline: CSV rows -> Document(page_content, metadata) -> embeddings ->
Chroma. page_content carries rich natural-language text so semantic
similarity search works well; metadata carries the structured fields so a
retriever could later filter on them (duration, category, etc.) in
addition to similarity.
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DIR
from src.data.data_loader import load_activities
from src.embeddings.embedding_service import get_embedding_function

COLLECTION_NAME = "activities"


def build_document(activity: dict) -> Document:
    """Turn one activity dict into a Document.

    page_content is a natural-language paragraph describing the activity so
    it matches well against natural-language retrieval queries (e.g. "a
    45-minute indoor social activity for someone who enjoys teaching").
    """
    interests = ", ".join(activity["interests"])
    purpose = ", ".join(activity["purpose"])
    tags = ", ".join(activity["tags"])
    instructions = " ".join(activity["instructions"])

    page_content = (
        f"{activity['title']}. {activity['description']} "
        f"This is a {activity['duration_minutes']}-minute {activity['location']} "
        f"activity with {activity['energy_level']} energy, suitable for a "
        f"{activity['skill_level']} skill level. "
        f"It is good for someone interested in {interests}. "
        f"Purpose: {purpose}. "
        f"Category: {activity['category']} ({activity['subcategory']}). "
        f"Materials needed: {activity['required_materials']}. "
        f"Steps: {instructions} "
        f"Tags: {tags}."
    )

    metadata = {
        "id": activity["id"],
        "title": activity["title"],
        "category": activity["category"],
        "subcategory": activity["subcategory"],
        "duration_minutes": activity["duration_minutes"],
        "energy_level": activity["energy_level"],
        "location": activity["location"],
        "skill_level": activity["skill_level"],
        "age_suitability": activity["age_suitability"],
        "interests": interests,
        "purpose": purpose,
        "required_materials": activity["required_materials"],
        "safety_notes": activity["safety_notes"],
    }
    return Document(page_content=page_content, metadata=metadata)


def build_documents(activities: list[dict]) -> list[Document]:
    return [build_document(a) for a in activities]


def ingest(csv_path: str | None = None, persist_dir: str = CHROMA_DIR) -> Chroma:
    """Load activities, embed them, and persist to a Chroma collection.
    Re-running this rebuilds the collection from scratch (idempotent)."""
    activities = load_activities(csv_path) if csv_path else load_activities()
    documents = build_documents(activities)
    ids = [a["id"] for a in activities]

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Clear any existing collection so re-ingestion doesn't duplicate rows.
    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)

    vectorstore.add_documents(documents=documents, ids=ids)
    return vectorstore


if __name__ == "__main__":
    store = ingest()
    count = store._collection.count()
    print(f"Ingested {count} activities into Chroma at {CHROMA_DIR}")
