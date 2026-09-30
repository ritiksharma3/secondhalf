"""Generates the teacher-labeled training set for the student classifier.

Builds a diverse set of user-like queries (varying phrasing, duration,
family members, moods, activity types), sends each through the teacher LLM
(see teacher.py) once, and appends the labeled examples to
data/teacher_training_data.jsonl.

Checkpointed: queries whose exact text already has a line in the JSONL are
skipped, so a rerun after a rate-limit error resumes instead of re-paying
for every example from scratch.
"""
import json
import time

from groq import RateLimitError

from src.config import TEACHER_DATA_JSONL
from src.distillation.teacher import TeacherLabelError, get_teacher_llm, label_query

# --- Seed query generation -------------------------------------------------
# Real user requests are naturally template-shaped ("I have X minutes and
# want to Y") - unlike the activity dataset (which needed to read as
# individually authored), template expansion here mirrors how people
# actually phrase these requests and gives the student varied but
# realistic training signal.

DURATIONS = ["15 minutes", "20 minutes", "30 minutes", "45 minutes", "an hour", "a couple of hours"]

FAMILY_TEMPLATES = [
    "Give me something I can do with my {person} for {duration}.",
    "I want to spend {duration} with my {person}.",
    "What's a good activity to do with my {person} this weekend?",
]
FAMILY_PEOPLE = ["grandson", "granddaughter", "grandchildren", "daughter", "son", "niece and nephew"]

USEFUL_TEMPLATES = [
    "I have {duration} and want to do something useful at home.",
    "I want to fix or organize something around the house, I have {duration}.",
    "Give me a productive task I can finish in {duration}.",
]

LEARN_TEMPLATES = [
    "I want to learn something new today.",
    "Teach me something in {duration}.",
    "I'd like to pick up a new skill, I have {duration}.",
    "I'm curious about {topic}, what can I do about it in {duration}?",
]
LEARN_TOPICS = ["history", "a new language", "science", "cooking", "technology", "gardening"]

SOCIAL_TEMPLATES = [
    "I want something social to do, I have {duration}.",
    "I'd like to meet or talk to people, I have {duration} free.",
    "I'm feeling lonely and want some company for {duration}.",
]

CONTRIBUTE_TEMPLATES = [
    "I want to help someone today.",
    "I have {duration} and want to volunteer or do something for the community.",
    "How can I contribute to my neighbourhood in {duration}?",
]

TIRED_TEMPLATES = [
    "I feel tired today and don't want to go outside.",
    "I'm low on energy but still want to do something meaningful, {duration}.",
    "I don't feel like moving much today, what can I do at home for {duration}?",
]

OUTDOOR_TEMPLATES = [
    "I want to go outside and get some fresh air, {duration}.",
    "I feel energetic today, give me something active outdoors.",
    "I want a physical activity I can do outside in {duration}.",
]

CREATE_TEMPLATES = [
    "I have {duration} and want to create something with my hands.",
    "I feel like making or building something today.",
    "Give me a creative project for {duration}.",
]

MEMORY_TEMPLATES = [
    "I want to look back on old memories today.",
    "I have {duration} and want to organize or revisit old photos.",
    "I'd like to write down some memories from my life, {duration}.",
]

MINDFUL_TEMPLATES = [
    "I want some quiet time to reflect, {duration}.",
    "I need to relax and clear my head for {duration}.",
    "I want to do something peaceful and calm today.",
]

HOBBY_TEMPLATES = [
    "I enjoy gardening but I did it yesterday, what else can I do?",
    "I have {duration} free for a hobby I already enjoy.",
    "Suggest something fun for {duration}, nothing too serious.",
]

REPETITION_TEMPLATES = [
    "I already went for a walk this morning, what else can I do for {duration}?",
    "I don't want to repeat what I did yesterday, give me something for {duration}.",
    "I enjoy {topic} but did it yesterday, suggest something else for {duration}.",
]

PRACTICAL_MORE_TEMPLATES = [
    "Something needs fixing around the house, I have {duration}, what should I tackle?",
    "I want to tidy up and organize for {duration}.",
    "Give me a small home-improvement task for {duration}.",
]

MOOD_TEMPLATES = [
    "I'm feeling {mood} today, suggest something for {duration}.",
    "I'm in a {mood} mood, what should I do for {duration}?",
]
MOODS = ["nostalgic", "curious", "restless", "peaceful", "sociable", "reflective"]


def _fmt(templates: list[str], **choices) -> list[str]:
    """Exhaustively expand every {placeholder} in each template against its
    option pool (itertools.product), so a template using two placeholders
    with 6 options each yields all 36 combinations rather than a random
    handful - this is what gets seed-query count up into the hundreds."""
    import itertools
    import re

    out: list[str] = []
    for t in templates:
        keys = [k for k in choices if "{" + k + "}" in t]
        if not keys:
            out.append(t)
            continue
        pools = [choices[k] for k in keys]
        for combo in itertools.product(*pools):
            filled = t
            for key, value in zip(keys, combo):
                filled = filled.replace("{" + key + "}", value)
            out.append(filled)
    return out


def generate_seed_queries() -> list[str]:
    """Return a deduplicated list of ~350-450 user-like query strings
    spanning all 10 categories and a range of durations/moods."""
    queries: list[str] = []
    queries += _fmt(FAMILY_TEMPLATES, person=FAMILY_PEOPLE, duration=DURATIONS)
    queries += _fmt(USEFUL_TEMPLATES, duration=DURATIONS)
    queries += _fmt(LEARN_TEMPLATES, duration=DURATIONS, topic=LEARN_TOPICS)
    queries += _fmt(SOCIAL_TEMPLATES, duration=DURATIONS)
    queries += _fmt(CONTRIBUTE_TEMPLATES, duration=DURATIONS)
    queries += _fmt(TIRED_TEMPLATES, duration=DURATIONS)
    queries += _fmt(OUTDOOR_TEMPLATES, duration=DURATIONS)
    queries += _fmt(CREATE_TEMPLATES, duration=DURATIONS)
    queries += _fmt(MEMORY_TEMPLATES, duration=DURATIONS)
    queries += _fmt(MINDFUL_TEMPLATES, duration=DURATIONS)
    queries += _fmt(HOBBY_TEMPLATES, duration=DURATIONS)
    queries += _fmt(REPETITION_TEMPLATES, duration=DURATIONS, topic=LEARN_TOPICS)
    queries += _fmt(PRACTICAL_MORE_TEMPLATES, duration=DURATIONS)
    queries += _fmt(MOOD_TEMPLATES, duration=DURATIONS, mood=MOODS)

    # de-dupe while preserving order
    seen = set()
    deduped = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def _load_already_labeled(path: str) -> set[str]:
    labeled = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                labeled.add(json.loads(line)["query"])
    except FileNotFoundError:
        pass
    return labeled


def generate_training_data(
    out_path: str = TEACHER_DATA_JSONL, limit: int | None = None, sleep_seconds: float = 0.0
) -> tuple[int, int]:
    """Label every seed query not already in out_path via the teacher LLM,
    appending each success immediately (checkpointing). Returns
    (num_written, num_failed)."""
    queries = generate_seed_queries()
    if limit:
        queries = queries[:limit]

    already = _load_already_labeled(out_path)
    todo = [q for q in queries if q not in already]

    llm = get_teacher_llm()
    written, failed = 0, 0

    with open(out_path, "a", encoding="utf-8") as f:
        for i, query in enumerate(todo, 1):
            for attempt in range(5):
                try:
                    label = label_query(query, llm=llm)
                    record = {"query": query, **label}
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    written += 1
                    break
                except RateLimitError:
                    wait = 15 * (attempt + 1)
                    print(f"[{i}/{len(todo)}] rate limited, waiting {wait}s...")
                    time.sleep(wait)
                except TeacherLabelError as e:
                    print(f"[{i}/{len(todo)}] SKIPPED (bad label): {query!r} - {e}")
                    failed += 1
                    break
            if sleep_seconds:
                time.sleep(sleep_seconds)
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}] labeled so far: {written}, failed: {failed}")

    return written, failed


if __name__ == "__main__":
    written, failed = generate_training_data()
    print(f"Done. Newly written: {written}, failed: {failed}")
