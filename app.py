"""SecondHalf - Streamlit UI.

Run: streamlit run app.py

The app is a short, one-way flow on purpose (spec section 19): the goal is
to get the user OFF the screen, not keep them on it.

    Profile -> Today's request -> Recommendation -> Phone away -> Feedback

All pipeline logic lives in src/recommendation/recommender.py; this file
only collects input, calls get_recommendation(), and renders the result.
"""
import json
import os

import streamlit as st

from src.config import CHROMA_DIR, STUDENT_MODEL_DIR
from src.data.data_loader import load_activities
from src.distillation.student import StudentNotTrainedError
from src.distillation.student import predict as student_predict
from src.embeddings.embedding_service import get_embedding_function
from src.profile import profile_store as ps
from src.rag.chain import InvalidRecommendationError
from src.rag.prompts import format_retrieved_context
from src.recommendation.recommender import get_recommendation

DEMO_REQUEST = "I want to do something useful and preferably with another person."

# UI button -> dataset category. "Surprise me" leaves it to the student model.
CATEGORY_CHOICES = {
    "✨ Surprise me": None,
    "📚 Learn": "learn",
    "🎨 Create": "create",
    "🤝 Connect": "connect",
    "🙏 Contribute": "contribute",
    "🔧 Useful": "practical_useful",
    "🌳 Outside": "physical_outdoor",
}
ALL_CATEGORIES = [
    "learn", "create", "connect", "contribute", "physical_outdoor",
    "hobbies", "family", "memory", "practical_useful", "mindfulness_reflection",
]
ENERGY_LEVELS = ["low", "medium", "high"]
SOCIAL_LEVELS = ["low", "medium", "high"]

st.set_page_config(page_title="SecondHalf", page_icon="🌿", layout="centered")

st.markdown(
    """
    <style>
      .flow {display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; font-size:.85rem; opacity:.8; margin-bottom:1rem}
      .flow span.step {padding:.15rem .6rem; border-radius:999px; border:1px solid rgba(128,128,128,.4)}
      .flow span.on {background:#2e7d32; color:white; border-color:#2e7d32}
      .mission {padding:1rem 1.2rem; border-radius:.8rem; background:rgba(46,125,50,.12); border-left:5px solid #2e7d32; font-size:1.1rem}
      div.stButton > button[kind="primary"] {font-size:1.15rem; padding:.7rem 1.4rem}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- startup ----------

def ensure_artifacts() -> None:
    """Build the Chroma index and student model if they're missing.

    Both are git-ignored, so a fresh cloud deploy (e.g. Streamlit Community
    Cloud) starts without them. Neither step needs an API key: ingestion
    embeds the CSV locally and training uses the committed teacher labels.
    """
    if not (os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR)):
        from src.rag.ingestion import ingest
        ingest()
    model_files = ("category_pipeline.joblib", "intent_pipeline.joblib")
    if not all(os.path.exists(os.path.join(STUDENT_MODEL_DIR, f)) for f in model_files):
        from src.distillation.train_student import train
        train()


@st.cache_resource(show_spinner="Getting SecondHalf ready (first start only, ~30 seconds)…")
def warm_up() -> bool:
    """Load the embedding model and student once per server process.

    Importing sentence-transformers + loading MiniLM takes ~25s here; doing
    it on page load instead of on the first "Find my activity" click keeps
    the demo's first recommendation to a couple of seconds. Failures are
    swallowed - the real call reports them with a friendly message.
    """
    try:
        ensure_artifacts()
        get_embedding_function().embed_query("warm up")
        student_predict("warm up")
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------- state helpers ----------

def init_state() -> None:
    defaults = {
        "step": "today" if ps.load_profile() else "profile",
        "result": None,
        "free_text": "",
        "editing_profile": False,
        "last_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go(step: str) -> None:
    st.session_state.step = step


@st.cache_data
def interest_options() -> list[str]:
    return sorted({i for a in load_activities() for i in a["interests"]})


def flow_bar(current: str) -> None:
    steps = [("today", "Choose"), ("recommendation", "Recommendation"), ("offline", "Phone away"), ("feedback", "Feedback")]
    html = " → ".join(
        f'<span class="step {"on" if key == current else ""}">{label}</span>' for key, label in steps
    )
    st.markdown(f'<div class="flow">{html}</div>', unsafe_allow_html=True)


def friendly_error(e: Exception) -> str:
    """Map pipeline failures to messages a user can act on (spec section 22).
    Never includes the API key - exception text from providers is not shown."""
    name = type(e).__name__
    if isinstance(e, StudentNotTrainedError):
        return "The activity classifier hasn't been trained yet. Run `python -m src.distillation.train_student` first."
    if isinstance(e, FileNotFoundError):
        return "The activity dataset is missing. Check that data/activities.csv exists."
    if isinstance(e, RuntimeError) and "GROQ_API_KEY" in str(e):
        return "No Groq API key found. Add GROQ_API_KEY to your .env file and restart the app."
    if isinstance(e, InvalidRecommendationError) or "BadRequest" in name:
        return "The AI gave an answer we couldn't read. Please try again."
    if isinstance(e, ValueError) and "No matching activities" in str(e):
        return str(e)
    if "Timeout" in name:
        return "The recommendation service took too long to respond. Please try again in a moment."
    if "Authentication" in name or "PermissionDenied" in name:
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env file."
    if "Connection" in name:
        return "Couldn't reach the recommendation service. Check your internet connection."
    if "RateLimit" in name:
        return "Too many requests to the AI service right now. Wait a minute and try again."
    if "chroma" in name.lower() or "chroma" in str(e).lower():
        return "The activity database couldn't be opened. Run `python -m src.rag.ingestion` to rebuild it."
    return f"Something went wrong ({name}). Please try again."


# ---------- sidebar: profile summary, history, demo ----------

def sidebar() -> None:
    profile = ps.load_profile()
    with st.sidebar:
        st.header("🌿 SecondHalf")
        if st.button("Load demo profile (Mr. Sharma)", use_container_width=True):
            ps.load_demo()
            st.session_state.free_text = DEMO_REQUEST
            st.session_state.result = None
            go("today")
            st.rerun()

        if profile:
            st.subheader(f"{profile.name}, {profile.age}")
            st.caption("Interests: " + (", ".join(profile.interests) or "—"))
            st.caption(f"Usually: {profile.typical_energy} energy · {profile.preferred_duration} min · "
                       f"{profile.social_preference} social")
            if st.button("Edit profile", use_container_width=True):
                go("profile")
                st.rerun()

        history = ps.load_history()
        if history:
            st.subheader("Recent activities")
            for entry in history[:5]:
                feedback = ps.FEEDBACK_LABELS.get(entry.feedback, "not rated yet")
                st.markdown(f"**{ps.days_ago_label(entry.date)}**  \n{entry.title} — {feedback}")
            avoid = ps.titles_to_avoid(history)
            if avoid:
                st.caption("Skipped today to avoid repetition: " + "; ".join(avoid))


# ---------- screen 1: profile ----------

def screen_profile() -> None:
    st.subheader("Tell us a little about yourself")
    current = ps.load_profile() or ps.UserProfile()
    options = sorted(set(interest_options()) | set(current.interests))
    with st.form("profile"):
        name = st.text_input("Name", current.name)
        age = st.number_input("Age", min_value=40, max_value=110, value=current.age)
        interests = st.multiselect("Interests", options, default=current.interests)
        preferred = st.multiselect("Preferred activity types", ALL_CATEGORIES, default=current.preferred_categories)
        energy = st.select_slider("Typical energy", ENERGY_LEVELS, value=current.typical_energy)
        duration = st.slider("Usual available time (minutes)", 10, 120, current.preferred_duration, step=5)
        social = st.select_slider("How much do you enjoy company?", SOCIAL_LEVELS, value=current.social_preference)
        if st.form_submit_button("Save profile", type="primary"):
            if not name.strip():
                st.error("Please enter a name.")
                return
            ps.save_profile(ps.UserProfile(
                name=name.strip(), age=int(age), interests=interests, preferred_categories=preferred,
                typical_energy=energy, preferred_duration=duration, social_preference=social,
            ))
            go("today")
            st.rerun()


# ---------- screen 2: today's activity ----------

def screen_today() -> None:
    profile = ps.load_profile()
    if not profile:
        go("profile")
        st.rerun()
    flow_bar("today")
    st.subheader(f"Namaste, {profile.name}. What would you like to do today?")

    choice = st.radio("Type of activity", list(CATEGORY_CHOICES), horizontal=True, label_visibility="collapsed")
    col1, col2 = st.columns(2)
    duration = col1.slider("Available time (minutes)", 10, 120, profile.preferred_duration, step=5)
    energy = col2.select_slider("Energy level", ENERGY_LEVELS, value=profile.typical_energy)
    free_text = st.text_area("Tell us what you're in the mood for (optional)", key="free_text",
                             placeholder="e.g. something calm at home, or something with my grandson")

    if st.button("Find my activity", type="primary"):
        with st.spinner("Finding something good for you…"):
            try:
                st.session_state.result = get_recommendation(
                    free_text=free_text,
                    duration_minutes=duration,
                    energy_level=energy,
                    social_preference=profile.social_preference,
                    interests=profile.interests,
                    avoid_titles=ps.titles_to_avoid(ps.load_history()),
                    category_override=CATEGORY_CHOICES[choice],
                )
                st.session_state.last_error = None
                go("recommendation")
                st.rerun()
            except Exception as e:  # noqa: BLE001 - every failure gets a friendly message
                st.session_state.last_error = friendly_error(e)
    if st.session_state.last_error:
        st.error(st.session_state.last_error)


# ---------- screen 3: recommendation ----------

def technical_details(result) -> None:
    """Spec section 25: prove RAG is really happening."""
    with st.expander("🔍 Technical details (how this was chosen)"):
        pred = result.student_prediction
        st.markdown("**1. Student model prediction** (local classifier, no API call)")
        st.code(f"category = {pred.category}\nintent   = {pred.intent}")
        if result.category_used != pred.category:
            st.caption(f"You picked a category explicitly, so retrieval used **{result.category_used}** "
                       f"instead of the student's guess.")

        st.markdown("**2. Retrieval query** (sent to Chroma)")
        st.code(result.retrieval_query, language=None)

        st.markdown("**3. Top-k retrieved activities** (cosine similarity)")
        st.table([
            {"#": i, "activity": doc.metadata["title"], "category": doc.metadata["category"],
             "minutes": doc.metadata["duration_minutes"], "similarity": round(score, 3)}
            for i, (doc, score) in enumerate(result.retrieved, 1)
        ])

        st.markdown("**4. Context given to the LLM**")
        st.code(format_retrieved_context(result.retrieved), language=None)

        st.markdown("**5. LLM output** (validated against the Pydantic schema)")
        st.code(json.dumps(result.recommendation.model_dump(), indent=2, ensure_ascii=False), language="json")

        st.markdown("**6. Grounding check**")
        verdict = "✅ grounded" if result.is_grounded else "⚠️ NOT grounded"
        st.write(f"{verdict} — closest retrieved activity: *{result.grounded_match}*")


def screen_recommendation() -> None:
    result = st.session_state.result
    if result is None:
        go("today")
        st.rerun()
    rec = result.recommendation
    flow_bar("recommendation")

    st.header(rec.title)
    st.caption(f"{rec.category.replace('_', ' ').title()} · about {rec.duration_minutes} minutes")
    if not result.is_grounded:
        st.warning("This suggestion didn't closely match our activity library, so treat it with care.")

    st.markdown("#### Why this activity?")
    st.write(rec.reason)
    for point in rec.why_it_matches:
        st.markdown(f"- {point}")

    st.markdown("#### What to do")
    for i, step in enumerate(rec.instructions, 1):
        st.markdown(f"{i}. {step}")

    st.markdown("#### You'll need")
    st.write(", ".join(rec.materials) if rec.materials else "Nothing special.")

    st.markdown("#### 📵 Phone-free mission")
    st.markdown(f'<div class="mission">{rec.phone_free_message}</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns([2, 1])
    if col1.button("I'll do this", type="primary", use_container_width=True):
        matched = result.grounded_match if result.is_grounded else rec.title
        ps.add_activity(matched, result.category_used or rec.category)
        st.session_state.committed_title = matched
        go("offline")
        st.rerun()
    if col2.button("Change my request", use_container_width=True):
        go("today")
        st.rerun()

    technical_details(result)


# ---------- phone away + feedback ----------

def screen_offline() -> None:
    flow_bar("offline")
    rec = st.session_state.result.recommendation
    st.header("📵 Time to put the phone down")
    st.markdown(f'<div class="mission">{rec.phone_free_message}</div>', unsafe_allow_html=True)
    st.write("")
    st.write(f"Enjoy **{rec.title}**. This page will wait here - come back only when you're done.")
    if st.button("I'm back", type="primary"):
        go("feedback")
        st.rerun()


def screen_feedback() -> None:
    flow_bar("feedback")
    title = st.session_state.get("committed_title", "")
    st.header("How was it?")
    st.caption(title)
    cols = st.columns(3)
    for col, (key, label) in zip(cols, ps.FEEDBACK_LABELS.items()):
        if col.button(label, use_container_width=True):
            ps.set_feedback(title, key)
            go("done")
            st.rerun()


def screen_done() -> None:
    st.header("Thank you! 🌿")
    st.write("Your feedback helps us suggest better activities next time. "
             "That's all for now - enjoy the rest of your day away from the screen.")
    if st.button("Plan another activity"):
        st.session_state.result = None
        go("today")
        st.rerun()


# ---------- main ----------

warm_up()
init_state()
st.title("SecondHalf")
st.markdown("*Spend less time scrolling. Spend more time living.*")
sidebar()

{
    "profile": screen_profile,
    "today": screen_today,
    "recommendation": screen_recommendation,
    "offline": screen_offline,
    "feedback": screen_feedback,
    "done": screen_done,
}[st.session_state.step]()
