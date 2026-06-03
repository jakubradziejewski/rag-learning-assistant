from __future__ import annotations

import os
import random
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import streamlit as st
from backend.core.audio.summary import DETAILS, MODES, summarize, synthesize_speech
from backend.core.rag.embedder import embed_text
from backend.core.rag.llm import grade_answer
from backend.core.srs.fsrs_scheduler import due_date, is_due, review_card
from backend.core.srs.generator import generate_items_for_chunk
from backend.core.srs.json_store import load_state, save_state, upsert_items
from backend.core.storage.vector_store import clear_documents, search
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

DATA_PATH = Path("data/srs_state.json")
UPLOAD_DIR = Path("uploads")


st.set_page_config(page_title="RAG Learning Assistant", layout="wide")
st.title("RAG Learning Assistant")


if "queue_ids" not in st.session_state:
    st.session_state.queue_ids = []
if "queue_index" not in st.session_state:
    st.session_state.queue_index = 0
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []
if "quiz_started_at" not in st.session_state:
    st.session_state.quiz_started_at = None
if "quiz_question_started_at" not in st.session_state:
    st.session_state.quiz_question_started_at = None
if "quiz_graded" not in st.session_state:
    st.session_state.quiz_graded = False
if "listen_text" not in st.session_state:
    st.session_state.listen_text = ""
if "listen_audio" not in st.session_state:
    st.session_state.listen_audio = None


st.sidebar.header("Daily Session Settings")
available_minutes = st.sidebar.number_input("Minutes available today", min_value=5, value=30, step=5)
minutes_per_item = st.sidebar.number_input("Minutes per item", min_value=1, value=2, step=1)


upload_tab, flashcards_tab, session_tab, quiz_tab, listen_tab, ask_tab = st.tabs(
    ["Upload", "Flashcards", "Daily session", "Quiz", "Listen", "Ask"]
)


with upload_tab:
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if st.button("Reset stored items"):
        try:
            clear_documents()
            if DATA_PATH.exists():
                DATA_PATH.unlink()
            if UPLOAD_DIR.exists():
                for path in UPLOAD_DIR.glob("*"):
                    if path.is_file():
                        path.unlink()
            st.session_state.queue_ids = []
            st.session_state.queue_index = 0
            st.session_state.show_answer = False
            st.session_state.quiz_questions = []
            st.session_state.quiz_index = 0
            st.session_state.quiz_results = []
            st.session_state.quiz_started_at = None
            st.session_state.quiz_question_started_at = None
            st.session_state.quiz_graded = False
            st.session_state.listen_text = ""
            st.session_state.listen_audio = None
            st.success("Stored items and uploads cleared.")
        except OSError as exc:
            st.error(f"Reset failed: {exc}")

    if uploaded_file and st.button("Process PDF"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

        with st.spinner("Uploading and processing PDF..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/documents/upload",
                    files=files,
                    timeout=600.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                st.error(f"Upload failed: {exc}")
                st.stop()

        payload = response.json()
        st.success(
            "Stored {stored} chunks for {filename}.".format(
                stored=payload.get("chunks_stored", 0),
                filename=payload.get("filename", "uploaded PDF"),
            )
        )


with flashcards_tab:
    st.subheader("Generate flashcards")
    state = load_state(DATA_PATH)
    existing_items = len(state.get("items", {}))
    st.write(f"Current flashcards: {existing_items}")

    st.caption(
        "This will clear existing flashcards and regenerate from all stored embeddings."
    )

    if st.button("Generate"):
        with st.spinner("Fetching stored chunks..."):
            try:
                response = httpx.get(
                    f"{API_BASE_URL}/documents/chunks",
                    timeout=300.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                st.error(f"Failed to fetch chunks: {exc}")
                st.stop()

        payload = response.json()
        chunks = payload.get("chunks", [])

        if not chunks:
            st.warning("No chunks found. Upload a PDF first.")
        else:
            state = load_state(DATA_PATH)
            state["items"] = {}

            st.session_state.queue_ids = []
            st.session_state.queue_index = 0
            st.session_state.show_answer = False

            total = len(chunks)
            progress = st.progress(0)
            items: list[dict] = []

            for index, chunk in enumerate(chunks, start=1):
                doc_id = chunk.get("doc_id", "")
                items.extend(generate_items_for_chunk(doc_id, chunk))
                progress.progress(index / total)

            upsert_items(state, items)
            save_state(DATA_PATH, state)

            st.success(
                "Generated {items} flashcards from {chunks} chunks.".format(
                    items=len(items),
                    chunks=total,
                )
            )


with session_tab:
    st.subheader("Daily session")
    state = load_state(DATA_PATH)
    items = list(state.get("items", {}).values())

    if not items:
        st.info("No study items yet. Upload a PDF first.")
    else:
        max_items = int(available_minutes // minutes_per_item)
        today = date.today()
        due_items = [item for item in items if is_due(item["card"], today)]
        due_items.sort(key=lambda item: due_date(item["card"]))

        # Shuffle only within the same due date to keep priority ordering.
        shuffled_due_items: list[dict] = []
        bucket: list[dict] = []
        current_due = None
        for item in due_items:
            item_due = due_date(item["card"])
            if current_due is None:
                current_due = item_due
            if item_due != current_due:
                random.shuffle(bucket)
                shuffled_due_items.extend(bucket)
                bucket = [item]
                current_due = item_due
            else:
                bucket.append(item)

        if bucket:
            random.shuffle(bucket)
            shuffled_due_items.extend(bucket)

        due_items = shuffled_due_items

        st.write(f"Due today: {len(due_items)} | Planned: {min(len(due_items), max_items)}")

        if st.button("Start session"):
            st.session_state.queue_ids = [item["id"] for item in due_items[:max_items]]
            st.session_state.queue_index = 0
            st.session_state.show_answer = False

        queue_ids = st.session_state.queue_ids
        if queue_ids:
            current_index = st.session_state.queue_index

            if current_index >= len(queue_ids):
                st.success("Session complete.")
            else:
                current_id = queue_ids[current_index]
                current_item = state["items"][current_id]

                st.write(
                    f"Item {current_index + 1} of {len(queue_ids)} | "
                    f"Type: {current_item['type']}"
                )
                st.markdown(f"**Prompt:** {current_item['prompt']}")

                if st.button("Show answer"):
                    st.session_state.show_answer = True

                if st.session_state.show_answer:
                    st.markdown(f"**Answer:** {current_item['answer']}")

                rating_labels = [
                    ("1 Again", 1),
                    ("2 Hard", 2),
                    ("3 Good", 3),
                    ("4 Easy", 4),
                    ("5 Perfect", 5),
                ]
                rating_cols = st.columns(len(rating_labels))
                rating_clicked = None
                for index, (label, score) in enumerate(rating_labels):
                    if rating_cols[index].button(
                        label,
                        key=f"rate_{current_id}_{score}",
                        disabled=not st.session_state.show_answer,
                    ):
                        rating_clicked = score

                if rating_clicked is not None:
                    print(f"Submitting rating {rating_clicked} for item {current_id}")
                    current_item["card"] = review_card(current_item["card"], rating_clicked)
                    current_item["last_review"] = datetime.now(timezone.utc).isoformat()
                    state["items"][current_id] = current_item
                    save_state(DATA_PATH, state)

                    st.session_state.queue_index += 1
                    st.session_state.show_answer = False
                    st.rerun()

                if st.button("Remove this question"):
                    state["items"].pop(current_id, None)
                    save_state(DATA_PATH, state)

                    if current_id in st.session_state.queue_ids:
                        st.session_state.queue_ids.remove(current_id)

                    if st.session_state.queue_index >= len(st.session_state.queue_ids):
                        st.session_state.queue_index = max(0, len(st.session_state.queue_ids) - 1)

                    st.session_state.show_answer = False
                    st.rerun()
        else:
            st.info("No items queued. Click 'Start session' to begin.")


def _format_seconds(total_seconds: int) -> str:
    minutes, seconds = divmod(max(0, total_seconds), 60)
    return f"{minutes}:{seconds:02d}"


@st.fragment(run_every=1)
def _quiz_timer() -> None:
    started_at = st.session_state.quiz_started_at
    if started_at is None:
        return

    now = datetime.now(timezone.utc)
    st.markdown(f"**Total time:** {_format_seconds(int((now - started_at).total_seconds()))}")

    question_started_at = st.session_state.quiz_question_started_at
    if question_started_at is not None and not st.session_state.quiz_graded:
        elapsed = int((now - question_started_at).total_seconds())
        st.markdown(f"**This question:** {_format_seconds(elapsed)}")


with quiz_tab:
    st.subheader("Quiz")

    if not st.session_state.quiz_questions:
        num_questions = st.number_input("Number of questions", min_value=1, value=5, step=1)

        if st.button("Start quiz"):
            with st.spinner("Fetching stored chunks..."):
                try:
                    response = httpx.get(
                        f"{API_BASE_URL}/documents/chunks",
                        timeout=300.0,
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    st.error(f"Failed to fetch chunks: {exc}")
                    st.stop()

            chunks = response.json().get("chunks", [])

            if not chunks:
                st.warning("No chunks found. Upload a PDF first.")
            else:
                candidates = [chunk for chunk in chunks if len(chunk.get("text", "").split()) >= 6]
                random.shuffle(candidates)
                questions: list[dict] = []

                with st.spinner("Generating questions..."):
                    for chunk in candidates:
                        if len(questions) >= num_questions:
                            break
                        for item in generate_items_for_chunk(chunk.get("doc_id", ""), chunk):
                            questions.append(item)
                            if len(questions) >= num_questions:
                                break

                if not questions:
                    st.warning("Could not generate questions from the stored chunks.")
                else:
                    now = datetime.now(timezone.utc)
                    st.session_state.quiz_questions = questions
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_results = []
                    st.session_state.quiz_started_at = now
                    st.session_state.quiz_question_started_at = now
                    st.session_state.quiz_graded = False
                    st.rerun()

    elif st.session_state.quiz_index >= len(st.session_state.quiz_questions):
        results = st.session_state.quiz_results
        total_score = sum(result["score"] for result in results)
        total_possible = len(results)
        percentage = round(100 * total_score / total_possible) if total_possible else 0
        total_seconds = sum(result["seconds"] for result in results)

        st.success("Quiz complete.")
        st.markdown(f"**Score:** {total_score:g} / {total_possible} ({percentage}%)")
        st.markdown(f"**Total time:** {_format_seconds(total_seconds)}")

        for index, result in enumerate(results, start=1):
            with st.expander(f"Q{index}: {result['score']:g} | {_format_seconds(result['seconds'])}"):
                st.markdown(f"**Question:** {result['prompt']}")
                st.markdown(f"**Your answer:**\n\n{result['user_answer'] or '(empty)'}")
                st.markdown(f"**Reference:**\n\n{result['reference']}")
                st.markdown(f"**Feedback:**\n\n{result['feedback']}")

        if st.button("New quiz"):
            st.session_state.quiz_questions = []
            st.session_state.quiz_index = 0
            st.session_state.quiz_results = []
            st.session_state.quiz_started_at = None
            st.session_state.quiz_question_started_at = None
            st.session_state.quiz_graded = False
            st.rerun()

    else:
        current_index = st.session_state.quiz_index
        current = st.session_state.quiz_questions[current_index]

        main_col, side_col = st.columns([3, 1])

        with side_col:
            _quiz_timer()

        with main_col:
            st.write(f"Question {current_index + 1} of {len(st.session_state.quiz_questions)}")
            st.markdown(f"**{current['prompt']}**")

            user_answer = st.text_area("Your answer", key=f"quiz_answer_{current_index}")

            if not st.session_state.quiz_graded:
                if st.button("Submit answer"):
                    with st.spinner("Grading..."):
                        query_embedding = embed_text(current["prompt"])
                        context_chunks = [
                            result["text"] for result in search(query_embedding, n_results=3)
                        ]
                        grade = grade_answer(
                            current["prompt"],
                            user_answer,
                            current["answer"],
                            context_chunks,
                        )

                    elapsed = int(
                        (datetime.now(timezone.utc) - st.session_state.quiz_question_started_at).total_seconds()
                    )
                    st.session_state.quiz_results.append(
                        {
                            "prompt": current["prompt"],
                            "reference": current["answer"],
                            "user_answer": user_answer,
                            "score": grade["score"],
                            "feedback": grade["feedback"],
                            "seconds": elapsed,
                        }
                    )
                    st.session_state.quiz_graded = True
                    st.rerun()
            else:
                result = st.session_state.quiz_results[-1]
                st.markdown(f"**Score:** {result['score']:g}")
                st.markdown(f"**Feedback:**\n\n{result['feedback']}")
                st.markdown(f"**Reference:**\n\n{result['reference']}")

                if st.button("Next question"):
                    st.session_state.quiz_index += 1
                    st.session_state.quiz_question_started_at = datetime.now(timezone.utc)
                    st.session_state.quiz_graded = False
                    st.rerun()


with listen_tab:
    st.subheader("Listen")

    try:
        response = httpx.get(f"{API_BASE_URL}/documents/chunks", timeout=300.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"Failed to fetch documents: {exc}")
        st.stop()

    chunks = response.json().get("chunks", [])

    if not chunks:
        st.info("No documents yet. Upload a PDF first.")
    else:
        documents: dict[str, dict] = {}
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "")
            document = documents.setdefault(
                doc_id, {"filename": chunk.get("filename", ""), "chunks": []}
            )
            document["chunks"].append(chunk)

        labels = {}
        for doc_id, document in documents.items():
            label = document["filename"] or f"Document {doc_id[:8]}"
            labels[label] = doc_id

        selected = st.multiselect("Source documents", list(labels.keys()))
        detail = st.radio("Detail", list(DETAILS.keys()), index=1, horizontal=True)
        mode = st.radio("Study mode", list(MODES.keys()))
        voice = st.selectbox("Voice", ["alloy", "echo", "fable", "nova", "shimmer", "onyx"])

        if st.button("Generate audio"):
            if not selected:
                st.warning("Select at least one document.")
            else:
                parts = []
                for label in selected:
                    doc_chunks = sorted(
                        documents[labels[label]]["chunks"],
                        key=lambda chunk: chunk.get("chunk_index", 0),
                    )
                    parts.append("\n".join(chunk.get("text", "") for chunk in doc_chunks))
                material = "\n\n".join(parts)

                try:
                    with st.spinner("Summarizing..."):
                        summary = summarize(material, mode, detail)
                    with st.spinner("Generating audio..."):
                        audio = synthesize_speech(summary, voice)
                except Exception as exc:
                    st.error(f"Generation failed: {exc} (is OPENAI_API_KEY set?)")
                    st.stop()

                st.session_state.listen_text = summary
                st.session_state.listen_audio = audio

        if st.session_state.listen_audio:
            st.audio(st.session_state.listen_audio, format="audio/wav")
            st.download_button(
                "Download audio",
                st.session_state.listen_audio,
                file_name="summary.wav",
                mime="audio/wav",
            )
            with st.expander("Summary text"):
                st.write(st.session_state.listen_text)


with ask_tab:
    st.subheader("Ask the material")
    question = st.text_input("Question")
    n_results = st.number_input("Top results", min_value=1, value=5, step=1)

    if question and st.button("Ask"):
        payload = {
            "question": question,
            "n_results": int(n_results),
            "temperature": 0.0,
        }

        with st.spinner("Searching and answering..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/documents/query",
                    json=payload,
                    timeout=300.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                st.error(f"Query failed: {exc}")
                st.stop()

        data = response.json()
        st.markdown("**Answer**")
        st.write(data.get("answer", ""))

        with st.expander("Sources"):
            for source in data.get("sources", []):
                section = source.get("section", "")
                pages = source.get("pages", "")
                st.write(f"Section: {section} | Pages: {pages}")
                st.write(source.get("text", ""))
