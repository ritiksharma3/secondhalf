"""Prompt template for the final recommendation LLM call.

The core grounding instruction - "use the retrieved activities as source of
truth, don't invent unsupported ones" - lives here (spec section 13) since
it's the single most important line for reducing hallucination.
"""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are SecondHalf, an assistant that recommends ONE offline activity to a \
retired adult so they can put their phone away and do something meaningful.

You are given a list of retrieved candidate activities (the RAG context) and the user's \
situation. You MUST select and personalize ONE activity from the retrieved candidates below. \
Use the retrieved activities as the source of truth - do not invent an activity, title, \
material, or instruction that isn't supported by the retrieved context. You may lightly \
personalize the wording (e.g. "teach a neighbour's child" instead of "teach a child") but the \
core activity must match one of the retrieved candidates.

Pick the single best match for the user's stated time, energy, mood, and interests. Write a \
short, warm, specific reason it fits THIS person. The phone_free_message should be a brief, \
encouraging line telling them to put their phone away and start."""

HUMAN_TEMPLATE = """User situation:
- Available time: {duration_minutes} minutes
- Energy level: {energy_level}
- Wants: {free_text}
- Recently done (avoid repeating): {avoid_titles}

Retrieved candidate activities (source of truth - pick one of these):
{retrieved_context}

Recommend ONE activity from the candidates above, personalized for this user."""

recommendation_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
)


def format_retrieved_context(scored_docs: list[tuple]) -> str:
    """Render retrieved (Document, score) pairs as numbered candidate
    blocks for the prompt."""
    blocks = []
    for i, (doc, score) in enumerate(scored_docs, 1):
        m = doc.metadata
        blocks.append(
            f"{i}. Title: {m['title']}\n"
            f"   Category: {m['category']} | Duration: {m['duration_minutes']} min | "
            f"Energy: {m['energy_level']} | Location: {m['location']}\n"
            f"   Materials: {m['required_materials']}\n"
            f"   Description: {doc.page_content}\n"
            f"   Similarity score: {score:.3f}"
        )
    return "\n\n".join(blocks)
