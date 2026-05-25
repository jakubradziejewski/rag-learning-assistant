from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import streamlit as st
from backend.core.srs.fsrs_scheduler import due_date, is_due, review_card
from backend.core.srs.generator import generate_items_for_chunks
from backend.core.srs.json_store import load_state, save_state, upsert_items
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


st.sidebar.header("Daily Session Settings")
available_minutes = st.sidebar.number_input("Minutes available today", min_value=5, value=30, step=5)
minutes_per_item = st.sidebar.number_input("Minutes per item", min_value=1, value=2, step=1)


upload_tab, session_tab, ask_tab = st.tabs(["Upload", "Daily session", "Ask"])


with upload_tab:
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    max_chunks = st.number_input(
        "Max chunks per PDF for study items",
        min_value=1,
        value=20,
        step=1,
        help=(
            "Limits how many parsed chunks are returned for study item generation. "
            "Lower values create fewer items and speed up generation. "
            "All chunks are still embedded and stored for search."
        ),
    )

    if st.button("Reset stored items"):
        try:
            if DATA_PATH.exists():
                DATA_PATH.unlink()
            if UPLOAD_DIR.exists():
                for path in UPLOAD_DIR.glob("*"):
                    if path.is_file():
                        path.unlink()
            st.session_state.queue_ids = []
            st.session_state.queue_index = 0
            st.session_state.show_answer = False
            st.success("Stored items and uploads cleared.")
        except OSError as exc:
            st.error(f"Reset failed: {exc}")

    if uploaded_file and st.button("Process PDF"):
        params = {"include_chunks": "true", "max_chunks": str(int(max_chunks))}
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

        with st.spinner("Uploading and processing PDF..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/documents/upload",
                    params=params,
                    files=files,
                    timeout=600.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                st.error(f"Upload failed: {exc}")
                st.stop()

        payload = response.json()
        chunks = payload.get("chunks", [])
        if not chunks:
            st.error("No chunks returned from the backend.")
        else:
            doc_id = payload["doc_id"]
            items = generate_items_for_chunks(doc_id, chunks)

            state = load_state(DATA_PATH)
            upsert_items(state, items)
            save_state(DATA_PATH, state)

            st.success(
                "Stored {stored} chunks. Generated {items} study items from {chunks} chunks.".format(
                    stored=payload.get("chunks_stored", 0),
                    items=len(items),
                    chunks=len(chunks),
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

                rating = st.slider("Your rating (1-5)", min_value=1, max_value=5, value=3)

                if st.button("Submit rating"):
                    print(f"Submitting rating {rating} for item {current_id}")
                    current_item["card"] = review_card(current_item["card"], rating)
                    current_item["last_review"] = datetime.now(timezone.utc).isoformat()
                    state["items"][current_id] = current_item
                    save_state(DATA_PATH, state)

                    st.session_state.queue_index += 1
                    st.session_state.show_answer = False
                    st.rerun()
        else:
            st.info("No items queued. Click 'Start session' to begin.")


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
