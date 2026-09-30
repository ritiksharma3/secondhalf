"""Headless click-through of the Streamlit app (streamlit.testing.AppTest)
with the LLM faked: demo profile -> request -> recommendation -> commit ->
feedback stored."""
import os

from streamlit.testing.v1 import AppTest

from src.profile import profile_store as ps
from src.rag import chain

from tests.conftest import FakeLLM, make_recommendation

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def click(at: AppTest, label_prefix: str) -> None:
    next(b for b in at.button if b.label.startswith(label_prefix)).click().run()


def test_demo_flow_end_to_end(user_data, monkeypatch):
    rec = make_recommendation("Tutor an underprivileged child for free", "contribute")
    monkeypatch.setattr(chain, "get_recommendation_llm", lambda: FakeLLM(rec))

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert at.session_state.step == "profile"  # no profile yet

    click(at, "Load demo profile")
    assert at.session_state.step == "today"
    assert at.text_area[0].value.startswith("I want to do something useful")

    click(at, "Find my activity")
    assert not at.exception and not at.error
    assert at.session_state.step == "recommendation"
    assert at.header[0].value == rec.title

    click(at, "I'll do this")
    click(at, "I'm back")
    click(at, "😊")
    assert at.session_state.step == "done"

    latest = ps.load_history()[0]
    assert latest.feedback == ps.FEEDBACK_LOVED
    assert latest.title in ps.titles_to_avoid(ps.load_history())


def test_missing_key_shows_friendly_error(user_data, monkeypatch):
    def no_key():
        raise RuntimeError("GROQ_API_KEY is not set.")

    monkeypatch.setattr(chain, "get_recommendation_llm", no_key)
    ps.load_demo()
    at = AppTest.from_file(APP, default_timeout=120).run()
    click(at, "Find my activity")
    assert not at.exception
    assert "GROQ_API_KEY" in at.error[0].value
    assert at.session_state.step == "today"
