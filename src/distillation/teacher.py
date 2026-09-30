"""Teacher: calls the Groq LLM to produce structured intent/category/
energy/social labels for a user-like query.

This module is ONLY used offline during data generation (see
generate_training_data.py) - the live app never calls the teacher, which is
the whole point of distillation (see student.py, which the app actually
uses at request time).
"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import Settings, load_settings, require_groq_key

CATEGORIES = [
    "learn",
    "create",
    "connect",
    "contribute",
    "physical_outdoor",
    "hobbies",
    "family",
    "memory",
    "practical_useful",
    "mindfulness_reflection",
]

SYSTEM_PROMPT = f"""You label short requests from retired adults looking for an offline activity to do.

Given one user request, output ONLY a JSON object with exactly these fields:
- "intent": a short snake_case label describing what the person wants (e.g. "family_activity", "productive_activity", "learn_something", "social_connection", "physical_exercise", "creative_project", "relax_and_reflect", "help_others")
- "category": exactly one of {CATEGORIES}
- "energy": one of "low", "medium", "high" - how much physical/mental energy the request implies
- "social": one of "low", "medium", "high" - how much social interaction the request implies

Output ONLY the JSON object, no other text, no markdown fences."""


class TeacherLabelError(Exception):
    """Raised when the teacher's response isn't valid/parseable JSON."""


def get_teacher_llm(settings: Settings | None = None) -> ChatGroq:
    settings = settings or load_settings()
    api_key = require_groq_key(settings)
    return ChatGroq(model=settings.groq_model, api_key=api_key, temperature=0)


def label_query(query: str, llm: ChatGroq | None = None) -> dict:
    """Call the teacher LLM once and return the parsed label dict.
    Raises TeacherLabelError on malformed output rather than silently
    guessing - callers (generate_training_data.py) skip and log failures
    instead of polluting the training set with bad labels."""
    llm = llm or get_teacher_llm()
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    )
    text = response.content.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise TeacherLabelError(f"No JSON object found in teacher response: {text!r}")
    try:
        label = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise TeacherLabelError(f"Invalid JSON from teacher: {text!r}") from e

    required = {"intent", "category", "energy", "social"}
    if not required.issubset(label.keys()):
        raise TeacherLabelError(f"Missing fields in teacher label: {label!r}")
    if label["category"] not in CATEGORIES:
        raise TeacherLabelError(f"Unknown category in teacher label: {label!r}")

    return label
