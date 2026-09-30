"""Lightweight user profile + activity history, persisted as one JSON file.

Single-user local app, so a JSON file is enough - no database. This module
is the only place that reads/writes that file (spec section 8), and it owns
the repetition-avoidance rule (spec section 18): which titles the retriever
should skip given recent history and feedback.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from src.config import USER_DATA_JSON

FEEDBACK_LOVED = "loved"
FEEDBACK_OKAY = "okay"
FEEDBACK_DISLIKED = "disliked"
FEEDBACK_LABELS = {
    FEEDBACK_LOVED: "😊 Loved it",
    FEEDBACK_OKAY: "😐 It was okay",
    FEEDBACK_DISLIKED: "🙁 Didn't enjoy it",
}

# Where profile + history live. Resolved at call time (not as a default
# argument) so tests can point it at a temp file.
DATA_PATH = USER_DATA_JSON

# Anything done in the last N days is skipped by retrieval, so the same
# activity isn't suggested two days running.
RECENT_DAYS = 3


@dataclass
class UserProfile:
    name: str = ""
    age: int = 65
    interests: list[str] = field(default_factory=list)
    preferred_categories: list[str] = field(default_factory=list)
    typical_energy: str = "medium"
    preferred_duration: int = 45
    social_preference: str = "medium"


@dataclass
class HistoryEntry:
    title: str
    category: str
    date: str  # ISO date the user did it
    feedback: str | None = None  # one of FEEDBACK_*; None until rated


def demo_profile() -> UserProfile:
    """Mr. Sharma, the deterministic demo user from spec section 24."""
    return UserProfile(
        name="Mr. Sharma",
        age=67,
        interests=["mathematics", "teaching", "gardening", "social interaction"],
        preferred_categories=["connect", "learn", "contribute"],
        typical_energy="medium",
        preferred_duration=45,
        social_preference="high",
    )


def demo_history(today: date | None = None) -> list[HistoryEntry]:
    """Mr. Sharma gardened yesterday - so gardening should be avoided today."""
    today = today or date.today()
    return [
        HistoryEntry("Tend to the garden", "physical_outdoor", (today - timedelta(days=1)).isoformat(), FEEDBACK_LOVED),
        HistoryEntry("Take a brisk morning walk", "physical_outdoor", (today - timedelta(days=2)).isoformat(), FEEDBACK_OKAY),
        HistoryEntry("Organize a family photo album", "family", (today - timedelta(days=3)).isoformat(), FEEDBACK_LOVED),
    ]


def _resolve(path: str | None) -> str:
    return path or DATA_PATH


def _read(path: str | None) -> dict:
    path = _resolve(path)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(data: dict, path: str | None) -> None:
    with open(_resolve(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_profile(path: str | None = None) -> UserProfile | None:
    raw = _read(path).get("profile")
    return UserProfile(**raw) if raw else None


def save_profile(profile: UserProfile, path: str | None = None) -> None:
    data = _read(path)
    data["profile"] = asdict(profile)
    _write(data, path)


def load_history(path: str | None = None) -> list[HistoryEntry]:
    """Newest first."""
    entries = [HistoryEntry(**e) for e in _read(path).get("history", [])]
    return sorted(entries, key=lambda e: e.date, reverse=True)


def save_history(entries: list[HistoryEntry], path: str | None = None) -> None:
    data = _read(path)
    data["history"] = [asdict(e) for e in entries]
    _write(data, path)


def add_activity(title: str, category: str, path: str | None = None, today: date | None = None) -> None:
    """Record that the user committed to an activity ("I'll do this")."""
    entries = load_history(path)
    entries.append(HistoryEntry(title, category, (today or date.today()).isoformat()))
    save_history(entries, path)


def set_feedback(title: str, feedback: str, path: str | None = None) -> None:
    """Attach feedback to the most recent unrated entry with this title."""
    if feedback not in FEEDBACK_LABELS:
        raise ValueError(f"Unknown feedback {feedback!r}")
    entries = load_history(path)  # newest first
    for entry in entries:
        if entry.title == title and entry.feedback is None:
            entry.feedback = feedback
            break
    save_history(entries, path)


def load_demo(path: str | None = None) -> None:
    """Reset the stored profile + history to the Mr. Sharma demo state."""
    _write({"profile": asdict(demo_profile()), "history": [asdict(e) for e in demo_history()]}, path)


def titles_to_avoid(history: list[HistoryEntry], today: date | None = None) -> list[str]:
    """Repetition avoidance: skip anything done in the last RECENT_DAYS days,
    plus anything the user said they didn't enjoy (ever)."""
    cutoff = ((today or date.today()) - timedelta(days=RECENT_DAYS)).isoformat()
    avoid = [e.title for e in history if e.date >= cutoff or e.feedback == FEEDBACK_DISLIKED]
    return list(dict.fromkeys(avoid))  # dedupe, keep order


def days_ago_label(iso_date: str, today: date | None = None) -> str:
    days = ((today or date.today()) - date.fromisoformat(iso_date)).days
    return {0: "Today", 1: "Yesterday"}.get(days, f"{days} days ago")
