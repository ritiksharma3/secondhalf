# SecondHalf

**An AI-powered offline-life companion for retired adults.**
*Spend less time scrolling. Spend more time living.*

SecondHalf suggests **one** meaningful offline activity, suited to the person, their free time, their energy and their mood, and then asks them to put the phone away. It is a college mini-project that shows **Retrieval-Augmented Generation (RAG)**, **task-specific teacher-student knowledge distillation**, **structured LLM output**, and **LLM tracing** working together in one small, honest pipeline.

## Notebook version (start here)

The whole project is also available as **one Jupyter notebook: [`SecondHalf.ipynb`](SecondHalf.ipynb)**. It contains the dataset, embeddings, retrieval, teacher-student distillation with its evaluation, the grounded LLM recommendation, profile and history, sanity checks, and an interactive app (ipywidgets) that replaces the Streamlit screens. It is saved with its outputs, so it can be read without running anything.

To run it:

```bash
pip install -r requirements.txt ipykernel ipywidgets matplotlib
```

Open `SecondHalf.ipynb` in Jupyter or VS Code, select this environment as the kernel, put `GROQ_API_KEY` in `.env`, and choose **Run All**. Without a key, everything except the final LLM step still runs.

The notebook only needs the `data/` folder. It does not import `app.py` or `src/`; those remain as the multi-file Streamlit version described below.

---

## 1. Project overview

| | |
|---|---|
| Input | A short profile (interests, usual energy, time, how social they are) + today's request ("I want to do something useful, preferably with another person") |
| Output | One activity: title, why it fits, duration, step-by-step instructions, materials, and a "phone-free mission" |
| After | "I'll do this" → phone away → come back → rate it (😊 😐 🙁) → future suggestions avoid repeats and disliked activities |

## 2. Problem statement

Many retired adults have large blocks of unstructured time. Smartphones fill that time easily, with endless feeds and short videos that are built to keep people scrolling. The result is often loneliness, inactivity and a feeling that the day has passed without purpose.

## 3. Why this problem matters

- Retirement removes the daily structure and social contact that work provided.
- Purposeful activity (teaching, volunteering, hobbies, family time) is linked to better wellbeing in older adults.
- Most apps compete for attention. SecondHalf is built to **give attention back**: it has no feed, no chat and no streaks, and its main job is to end the session quickly.

## 4. Solution

A short, one-way flow:

```
Profile → Today's request → ONE recommendation → "I'll do this" → Phone away → Feedback
```

Behind the flow:

1. A **student classifier** (local and fast) reads the free-text request and predicts the activity category and intent.
2. A **retriever** searches a custom library of 200 activities stored in a **Chroma** vector database.
3. An **LLM** (Groq) picks and personalises **one** of the retrieved activities. It is not allowed to invent new ones.
4. **Pydantic** validates the LLM output, and a **grounding check** confirms the answer matches a retrieved activity.
5. Feedback and history are stored, so recent or disliked activities are not suggested again.

## 5. Architecture

```mermaid
flowchart TD
    UI[Streamlit UI<br/>profile + today's request] --> S[Student classifier<br/>TF-IDF + Logistic Regression<br/>~1 ms, no API]
    S -->|category, intent| Q[Build retrieval query<br/>+ metadata filter]
    H[(History & feedback<br/>data/user_data.json)] -->|titles to avoid| Q
    Q --> C[(Chroma vector DB<br/>200 activities<br/>MiniLM embeddings)]
    C -->|top-5 activities + similarity| P[Prompt: retrieved activities<br/>are the source of truth]
    P --> L[Groq LLM<br/>structured output]
    L --> V[Pydantic<br/>ActivityRecommendation]
    V --> G{Grounding check<br/>matches a retrieved title?}
    G --> UI
    UI -->|"I'll do this" + rating| H

    subgraph Offline, once
      T[Teacher LLM - Groq] -->|labels 392 queries| D[(teacher_training_data.jsonl)]
      D -->|train| S
    end

    LS[LangSmith] -. traces every step .- S & C & L & G
```

**How the category is used (soft vs hard):**
- If the user presses a category button (Learn, Create, Connect, …), only that category is searched (a **hard filter**).
- If the user picks **"Surprise me"**, the student's predicted category is only a **ranking boost** (+0.05 similarity). A strong match from another category can still win. For example, Mr. Sharma asks for "something useful with another person". The student predicts `practical_useful`, but a *teaching* activity matches his interests better and is chosen.

## 6. Technology stack

| Purpose | Tool |
|---|---|
| UI | Streamlit |
| LLM orchestration | LangChain (LCEL, `ChatPromptTemplate`, `with_structured_output`) |
| LLM provider | Groq (`openai/gpt-oss-20b` by default; configurable) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally) |
| Vector database | Chroma (persisted on disk, cosine similarity) |
| Student model | scikit-learn: TF-IDF (1–2 grams) + Logistic Regression |
| Structured output | Pydantic |
| Tracing | LangSmith (optional) |
| Tests | pytest + Streamlit `AppTest` |

## 7. RAG explanation

**RAG = Retrieval-Augmented Generation.** Before asking the LLM to answer, we *retrieve* relevant facts and put them in the prompt. Here the "facts" are activities from our own curated dataset.

The pipeline (`src/rag/`):

```
activities.csv → Documents (ingestion.py) → embeddings → Chroma
user context → natural-language query (retriever.py) → similarity search + metadata filter → top-5
top-5 → prompt (prompts.py) → Groq LLM → ActivityRecommendation (chain.py) → grounding check
```

- **Query building:** the query is a plain sentence ("An activity for a retired adult who enjoys teaching, mathematics and wants high social interaction and …"). Hard constraints such as **duration (±20 min)** and **category** go into a Chroma metadata filter instead of the query text. Stuffing them into the text made similarity worse in testing: cosine similarity for the ideal activity fell from 0.60 to 0.35.
- **Repetition avoidance:** titles done in the last 3 days, or rated "Didn't enjoy", are removed from the results.
- **Fallbacks:** if the strict filter finds nothing, the filter is relaxed step by step, so a request never comes back empty just because the duration didn't line up.
- **Grounding:** the prompt says the retrieved activities are the *source of truth*. Afterwards, `check_grounding` fuzzy-matches the recommended title against the retrieved titles (threshold 0.5). A light personalisation ("Teach a maths concept **to your grandson**") passes; an unsupported invention ("Join a pottery class") is flagged, and the UI shows a warning.

## 8. Knowledge distillation explanation

> We use **task-specific teacher-student knowledge transfer**. A stronger teacher LLM generates structured labels for user-intent examples, and a lightweight student classifier learns to reproduce these predictions for repeated classification tasks.

- **Teacher** (`src/distillation/teacher.py`): a Groq LLM. Given a user-style query, it returns JSON labels: `category`, `intent`, `energy`, `social`. It runs **offline only**.
- **Training data** (`generate_training_data.py`): 392 realistic queries built from templates ("Give me something I can do with my grandson for 20 minutes."), each labelled once by the teacher → `data/teacher_training_data.jsonl`. The script saves progress as it goes, so it can resume after a rate-limit error.
- **Student** (`train_student.py`, `student.py`): two TF-IDF + Logistic Regression pipelines, one for `category` (drives retrieval) and one for `intent` (shown in the technical panel). They are trained on an 80/20 stratified split with a fixed seed.

**Honesty note:** this is *pseudo-label* distillation. The student learns the teacher's **hard labels**, not its probability distributions (logits). Groq's chat API does not expose per-class probabilities for a JSON classification answer, so classic soft-label/logit knowledge distillation (Hinton et al.) was not feasible. We describe it as task-specific teacher-student distillation, not logit-level KD.

## 9. Why distillation is used

The category has to be predicted on **every** request. Calling a large LLM for that is slow and costs money per call. The student does the same repeated task locally:

| Measured on the dev machine | Latency |
|---|---|
| Student (average of 5 predictions) | **~1.3–1.7 ms** |
| Teacher (one Groq API call) | **~850–1,500 ms** (depends on network) |
| Speed-up | **~500–1,200×** across runs |

The division of work: the **student** handles the simple, repeated classification; the **LLM** is kept for the one hard step, writing a personalised, grounded recommendation.

**Student quality** (held-out test set, n = 79, `python -m src.evaluation.evaluate`):

| Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|
| 0.924 | 0.933 | 0.897 | 0.904 |

The most common confusions are `create` ↔ `hobbies` (2 of 6 test examples), which overlap naturally, and `physical_outdoor`.

## 10. Dataset description

`data/activities.csv` holds **200 hand-designed activities**: 20 in each of 10 categories (`learn`, `create`, `connect`, `contribute`, `physical_outdoor`, `hobbies`, `family`, `memory`, `practical_useful`, `mindfulness_reflection`). They are written for retired adults in India: teaching grandchildren, temple walks, identifying ragas, tutoring, composting, organising family photos, repairing household items, and so on. Most need no smartphone.

Fields: `id, title, description, category, subcategory, duration_minutes, energy_level, location, required_materials, skill_level, interests, age_suitability, purpose, instructions, safety_notes, tags`. List fields (`interests`, `purpose`, `instructions`, `tags`) are stored as JSON strings. `src/data/data_loader.py` is the only code that parses them.

## 11. Folder structure

```
secondhalf/
├── app.py                          # Streamlit UI (all screens)
├── requirements.txt
├── .env.example                    # copy to .env
├── data/
│   ├── activities.csv              # 200-activity custom dataset
│   ├── teacher_training_data.jsonl # 392 teacher-labelled queries
│   ├── evaluation_questions.json   # spec's 8 evaluation questions
│   └── user_data.json              # profile + history (created at runtime, git-ignored)
├── models/student_classifier/      # trained student (git-ignored, rebuild with train_student)
├── chroma_db/                      # vector index (git-ignored, rebuild with ingestion)
├── docs/TECHNICAL_EXPLANATION.md   # viva preparation notes
├── src/
│   ├── config.py                   # ALL settings/.env loading lives here
│   ├── data/                       # data_generator.py (made the CSV), data_loader.py
│   ├── embeddings/                 # local MiniLM embedding model (cached)
│   ├── rag/                        # ingestion, retriever, prompts, chain (+ grounding)
│   ├── distillation/               # teacher, generate_training_data, train_student, student
│   ├── recommendation/             # recommender.py: runs the whole pipeline
│   ├── profile/                    # profile_store.py: profile, history, repetition rules
│   └── evaluation/                 # evaluate.py: metrics + latency
└── tests/                          # 33 pytest tests, no API key needed
```

## 12. Installation

Requires Python 3.12.

```bash
cd secondhalf
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The first run downloads the MiniLM embedding model (~90 MB) once. Each server start then spends ~30 s loading it; the app does this on page load so the first click is fast. **Before demo day, run the app once while online** so the model is cached.

## 13. Environment variables

Copy `.env.example` to `.env` and fill it in. Never commit `.env`; it is git-ignored.

| Variable | Required? | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes, for recommendations and teacher labelling | Free key at https://console.groq.com/keys |
| `GROQ_MODEL` | No (default `openai/gpt-oss-20b`) | Change the LLM without touching code |
| `LANGCHAIN_API_KEY` | Only for tracing | LangSmith key |
| `LANGCHAIN_TRACING_V2` | No (default `false`) | `true` to send traces |
| `LANGCHAIN_PROJECT` | No (default `secondhalf`) | LangSmith project name |

Retrieval, the student model and the tests all work **without any key**. Only the final recommendation needs `GROQ_API_KEY`, and it is checked when that step runs, with a clear message if it's missing.

## 14. Running the application

```bash
streamlit run app.py
```

Opens at http://localhost:8501. If this is a fresh clone, run steps 16 and 17 first (the index and model are git-ignored).

## 15. Generating teacher data (needs `GROQ_API_KEY`)

```bash
python -m src.distillation.generate_training_data
```

Labels any queries that don't have labels yet and appends them to `data/teacher_training_data.jsonl`. Safe to re-run: labelled queries are skipped. The labelled file is already included, so this step is optional.

## 16. Training the student model (no key needed)

```bash
python -m src.distillation.train_student
```

Trains both pipelines and writes them, plus the fixed train/test split, to `models/student_classifier/`.

## 17. Building the Chroma index (no key needed)

```bash
python -m src.rag.ingestion
```

Embeds all 200 activities into `chroma_db/`. Re-running rebuilds the index from scratch without duplicates.

## 18. Running evaluation

```bash
python -m src.evaluation.evaluate     # accuracy, precision, recall, F1, confusion matrix, latency
pytest tests/ -v                      # 33 tests: dataset, retrieval, grounding, pipeline, profile, UI flow
```

The tests replace the LLM with a fake, so they are deterministic and need no key. `tests/test_app.py` clicks through the real Streamlit app headlessly.

## 19. LangSmith setup

1. Create a free account at https://smith.langchain.com and create an API key.
2. In `.env`: `LANGCHAIN_API_KEY=<key>`, `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT=secondhalf`.
3. Run the app and make one recommendation. In LangSmith → project **secondhalf**, each request appears as one trace:

```
secondhalf_pipeline
├── student_classifier     (predicted category / intent)
├── rag_retrieval          (query + retrieved documents)
├── recommendation_chain
│   └── ChatGroq → PydanticToolsParser
└── grounding_check
```

With tracing off, the `@traceable` decorators do nothing, and the app works the same.

## 20. Demo instructions (8 minutes)

1. `streamlit run app.py` and open the browser.
2. **Sidebar → "Load demo profile (Mr. Sharma)"**: 67 years old; likes mathematics, teaching, gardening and company; 45 min; medium energy. The history shows **gardening yesterday**.
3. The request box is pre-filled: *"I want to do something useful and preferably with another person."* Leave **"✨ Surprise me"** selected → **Find my activity**.
4. Result: a **tutoring/teaching** activity, not gardening. Point out that the history was used.
5. Open **🔍 Technical details**: student prediction (`practical_useful`), retrieval query, top-5 with similarity scores, the exact context sent to the LLM, the validated JSON, and grounding ✅.
6. Click **I'll do this** → the phone-away screen → **I'm back** → **😊 Loved it**. The sidebar history updates, and that activity is now skipped for 3 days.
7. Optional: show the same request as a trace in LangSmith.
8. Optional: pick **🌳 Outside** to show that an explicit category is a hard filter.

**Before presenting:** start the app and open the page **at least a minute early**. The first page load after a server start takes ~30 s ("Getting SecondHalf ready…") because it loads the embedding model. After that, every recommendation takes ~2 s.

Demo safety: the LLM uses `temperature=0.3`, so titles vary a little in wording, but retrieval is deterministic. Groq occasionally rejects a malformed tool call (HTTP 400); the chain retries that once automatically.

## 21. Limitations

- **Imbalanced teacher data:** only 13 of 392 labelled queries are `contribute`, so the student classifies "I want to help someone" as `mindfulness_reflection`. In *Surprise me* mode the category is only a boost, which limits the damage, but more `contribute` examples are the real fix.
- **Hard-label distillation only:** no soft labels or logits (see §8).
- **Template-generated queries:** the student has seen template-style phrasing. Very unusual phrasing may be classified less accurately than the 92% test score suggests.
- **Fuzzy title grounding:** it checks the *title*, not every instruction the LLM writes. Heavy rewording of a valid title could be flagged falsely, and invented steps under a valid title would not be caught.
- **Single user, local JSON storage:** no accounts, no multi-device sync.
- **Energy level** is passed to the LLM but not used as a retrieval filter.
- Needs internet for the Groq call (and once for the embedding model download).

## 22. Future improvements

- Generate more teacher data for weak categories (`contribute`, `physical_outdoor`) and re-evaluate.
- Soft-label distillation, using an open model that exposes class probabilities.
- A RAG evaluation script (retrieval relevance@k, grounding rate, repetition rate over the 8 evaluation questions).
- An offline demo mode with a cached recommendation, for when there's no internet.
- Use the energy level and indoor/outdoor preference as retrieval filters.
- Hindi / regional-language interface and voice input for users who find typing difficult.
