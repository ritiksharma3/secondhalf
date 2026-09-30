from datetime import date, timedelta

import pytest

from src.profile import profile_store as ps

TODAY = date(2026, 9, 26)


def entry(title: str, days_ago: int, feedback: str | None = None) -> ps.HistoryEntry:
    return ps.HistoryEntry(title, "connect", (TODAY - timedelta(days=days_ago)).isoformat(), feedback)


def test_profile_round_trip(user_data):
    assert ps.load_profile() is None
    ps.save_profile(ps.demo_profile())
    assert ps.load_profile() == ps.demo_profile()


def test_recent_activities_are_avoided():
    history = [entry("Tend to the garden", 1), entry("Old walk", 10)]
    assert ps.titles_to_avoid(history, today=TODAY) == ["Tend to the garden"]


def test_disliked_activities_are_avoided_forever():
    history = [entry("Crossword", 30, ps.FEEDBACK_DISLIKED), entry("Chess", 30, ps.FEEDBACK_LOVED)]
    assert ps.titles_to_avoid(history, today=TODAY) == ["Crossword"]


def test_commit_then_feedback_is_stored(user_data):
    ps.add_activity("Teach a mathematics concept", "connect", today=TODAY)
    assert ps.load_history()[0].feedback is None

    ps.set_feedback("Teach a mathematics concept", ps.FEEDBACK_LOVED)
    saved = ps.load_history()[0]
    assert (saved.title, saved.feedback) == ("Teach a mathematics concept", "loved")


def test_unknown_feedback_rejected(user_data):
    with pytest.raises(ValueError):
        ps.set_feedback("anything", "meh")


def test_demo_state_avoids_gardening(user_data):
    ps.load_demo()
    assert ps.load_profile().name == "Mr. Sharma"
    assert "Tend to the garden" in ps.titles_to_avoid(ps.load_history())


def test_days_ago_label():
    assert ps.days_ago_label(TODAY.isoformat(), today=TODAY) == "Today"
    assert ps.days_ago_label((TODAY - timedelta(days=1)).isoformat(), today=TODAY) == "Yesterday"
    assert ps.days_ago_label((TODAY - timedelta(days=3)).isoformat(), today=TODAY) == "3 days ago"
