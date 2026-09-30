# SecondHalf: technical explanation (viva notes)

Plain-language answers to the questions examiners are most likely to ask. Every claim points to where it lives in the code, so you can open the file if asked.

---

## The pipeline in one breath

> The user types what they're in the mood for. A small **student model** we trained ourselves guesses the kind of activity (≈1 ms, no internet). We turn the user's profile and request into a search sentence, and **Chroma** finds the 5 most similar activities from **our own 200-activity dataset**. Those 5 go into a prompt, and a **Groq LLM** must pick and personalise **one of them**. **Pydantic** checks that the answer has the right shape, and a **grounding check** confirms it really came from those 5. **LangSmith** records every step.

Code path: `app.py` → `src/recommendation/recommender.py::get_recommendation` → `student.py` → `retriever.py` → `chain.py`.

---

## RAG

**What problem does RAG solve here?**
We want recommendations that are *realistic, safe, and suitable for retired adults in India*, from a list **we** control. RAG (Retrieval-Augmented Generation) first *retrieves* matching activities from our dataset, then asks the LLM to *generate* a personalised answer **using only those**.

**Why not simply ask the LLM to generate activities?**
- **Hallucination:** a bare LLM can invent activities that are unsafe, unrealistic for a 67-year-old, or need things the user doesn't have.
- **Control:** with RAG, the dataset is the source of truth. We decide what activities exist, their durations, materials and safety notes.
- **Explainability:** we can show *which* documents the answer came from, with similarity scores (the Technical Details panel).
- **Consistency:** retrieval is deterministic; the same request finds the same candidates.

**Why build a query sentence instead of searching the raw input?**
The embedding model compares *meanings*. A full sentence like "an activity for a retired adult who enjoys teaching and wants high social interaction" lands close to the right activities. Hard rules like "45 minutes" or "category = learn" are **filters**, not meaning, so they go into a Chroma metadata filter. Putting them into the sentence made similarity worse in testing (0.60 → 0.35) (`retriever.py::build_query`, `build_metadata_filter`).

---

## Embeddings

**What do they represent?**
An embedding is a list of numbers (here 384 of them) that represents the *meaning* of a text. Texts with similar meanings get vectors that point in similar directions. "Teach a child maths" and "tutor a student in arithmetic" end up close together even though they share almost no words.

**Why is semantic retrieval useful?**
Users don't phrase things the way the dataset is written. Keyword search would miss "something with my grandson" → "storytelling with grandchildren". Semantic search matches on meaning.

We use `all-MiniLM-L6-v2`, which runs **locally**: it's free, needs no API key, and gives the same result every time. Vectors are normalised, so we compare them by **cosine similarity** (1 = same meaning, 0 = unrelated) (`src/embeddings/embedding_service.py`).

---

## Chroma

**Why use a vector database?**
Comparing the query with every activity by hand works for 200 items but doesn't scale, and we'd have to handle storage and filtering ourselves. Chroma:
- stores each activity's **vector**, **text** and **metadata** together, on disk (`chroma_db/`);
- finds the nearest vectors quickly (HNSW index);
- supports **metadata filters** (`category = learn AND 25 ≤ duration ≤ 65`) in the same query.

Each activity becomes one LangChain `Document`: `page_content` is a rich paragraph used for meaning, and `metadata` holds the structured fields used for filtering (`src/rag/ingestion.py`).

---

## LangChain

**What is LangChain doing?**
It is the glue between our code and the LLM:
- `ChatPromptTemplate` builds the system and user messages from our template (`src/rag/prompts.py`).
- `ChatGroq` is the provider wrapper. Switching model is one line in `.env` (`GROQ_MODEL`).
- `.with_structured_output(ActivityRecommendation)` makes the LLM answer through a tool call that matches our Pydantic schema, then parses it into a Python object.
- `.with_retry(...)` retries once if Groq rejects a malformed tool call.
- `langchain-chroma` / `langchain-huggingface` connect the vector store and embedding model through the same `Document` interface.

---

## Knowledge distillation

**The honest one-sentence definition (say this):**
> We use **task-specific teacher-student knowledge transfer**. A stronger teacher LLM generates structured labels for user-intent examples, and a lightweight student classifier learns to reproduce these predictions for repeated classification tasks.

```
Teacher (Groq LLM, offline)  →  labels 392 example requests with category/intent/energy/social
Student (TF-IDF + LogReg)    →  learns to reproduce the teacher's category & intent labels
Student (at runtime)         →  handles the simple, repeated classification: ~1 ms, free, offline
LLM (at runtime)             →  handles the one complex step: the personalised, grounded recommendation
```

**Files:** `teacher.py` (labelling prompt), `generate_training_data.py` (392 template queries → `teacher_training_data.jsonl`), `train_student.py` (training), `student.py` (runtime), `evaluation/evaluate.py` (metrics).

**Results (held-out 20%, n = 79):** accuracy **0.924**, macro precision 0.933, recall 0.897, **F1 0.904**.
**Latency (measured across runs):** student **~1.3–1.7 ms** vs teacher **~850–1,500 ms**, so roughly **500–1,200× faster**, and free.

**Likely follow-up: "Is this *real* knowledge distillation?"**
It is distillation of the teacher's **decisions** (hard pseudo-labels), not its **probabilities**. Classic KD (Hinton, 2015) trains the student on the teacher's softened probability distribution (logits), which carries extra information such as "70% learn, 20% hobbies". Groq's chat API returns a JSON answer, not class probabilities, so logit-level KD wasn't possible. That's why we call it *task-specific teacher-student distillation* and don't claim more.

**"Why TF-IDF + Logistic Regression and not a neural network?"**
The task is short-text classification into 10 classes with ~400 examples. A linear model on word and bigram features is fast, explainable (you can inspect which words push towards which category), and reaches 92% accuracy. A neural model would add training cost for little gain at this data size.

**"Where does it go wrong?"**
`create` ↔ `hobbies` confusion (the categories overlap), and `contribute` is weak because only 13 of the 392 teacher examples were `contribute`. "I want to help someone" is misclassified. The fix is more teacher data for that class.

**"What if the student is wrong?"**
When the user picks "Surprise me", the student's category is only a **+0.05 ranking boost**, not a filter, so a strongly matching activity from another category can still win. When the user presses a category button, that choice is a hard filter and overrides the student (`retriever.py`, `CATEGORY_BOOST` in `config.py`).

---

## LangSmith

**Why is tracing useful?**
An LLM pipeline has several hidden steps. When an answer looks wrong, you need to know *which* step failed: did the student misclassify, did retrieval find poor candidates, or did the LLM ignore good ones? LangSmith records each request as a tree of steps with their inputs, outputs and timings:

```
secondhalf_pipeline → student_classifier → rag_retrieval → recommendation_chain (ChatGroq → PydanticToolsParser) → grounding_check
```

We add each step with the `@traceable` decorator. It's controlled by `LANGCHAIN_TRACING_V2` in `.env`; with tracing off, the decorators do nothing and the app is unchanged.

---

## Pydantic

**Why do structured outputs matter?**
The UI needs specific fields: title, reason, duration, a *list* of instructions, a *list* of materials, the phone-free message. Free text would have to be parsed with fragile string matching. `ActivityRecommendation` (`src/rag/chain.py`) defines the exact fields and types. The LLM is made to fill that schema, and Pydantic **validates** it. If validation fails, or nothing structured comes back, we raise `InvalidRecommendationError` and the UI shows "The AI gave an answer we couldn't read. Please try again" instead of crashing.

---

## Hallucination

**How does our retrieved-context constraint reduce hallucination?** Three layers:

1. **Prompt constraint:** the system prompt says the retrieved activities are the *source of truth*, that the model must pick **one of them**, and that it must not invent activities, materials or instructions. Light personalisation is allowed.
2. **Limited choice:** the LLM only ever sees 5 vetted candidates, not the whole world.
3. **Grounding check** (`chain.py::check_grounding`): after generation, we fuzzy-match the recommended title with every retrieved title.
   - Retrieved "Teach a mathematics concept" → LLM "Teach a mathematics concept to your grandson" ✅ accepted (personalisation)
   - Retrieved gardening + maths → LLM "Join a pottery class" ⚠️ flagged as unsupported, and the UI shows a warning

Both cases are tests in `tests/test_recommendation.py`.

**Known gap:** the check compares titles only. Invented *steps* under a valid title wouldn't be caught. Mitigation: the prompt forbids it, and the retrieved instructions are in the context for the model to reuse.

---

## Other questions you may get

| Question | Answer |
|---|---|
| How do you avoid repeating activities? | `profile_store.titles_to_avoid`: anything done in the last 3 days, plus anything ever rated "Didn't enjoy", is removed from the retrieval results. Mr. Sharma gardened yesterday, so gardening is skipped. |
| Where is feedback stored? | `data/user_data.json` (profile + history with ratings). One local JSON file, because it's a single-user app. |
| What happens without an API key? | Retrieval, the student model and all 33 tests still work. The recommendation step shows "No Groq API key found. Add GROQ_API_KEY to .env". |
| Why is the UI so minimal? | That's deliberate: the app's goal is to get the user **off** the screen. No feed, no chat, no badges. Recommendation → start → phone away → feedback. |
| How do you know RAG is really happening? | Open "🔍 Technical details": it shows the query, the top-5 retrieved activities with scores, the exact context sent to the LLM, and the grounding result. The LangSmith trace shows the same. |
| How is it tested? | 33 pytest tests: dataset shape, filters, soft vs hard category, avoid list, grounding accept/reject, schema rejection, missing key, full pipeline with a fake LLM, and a headless click-through of the Streamlit app. |
