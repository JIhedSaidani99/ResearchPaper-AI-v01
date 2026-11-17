# Run with:  streamlit run app.py

import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Literal

import pymupdf as fitz
import streamlit as st
from PIL import Image

import config
from src import indexing, library
from src.generation import AssistantTask
from src.pipeline import answer_question, format_answering_error

StatusColor = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(
    page_title="ResearchPaper AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
st.logo(
    str(APP_DIR / "assets" / "researchpaper-ai-logo.svg"),
    icon_image=str(APP_DIR / "assets" / "researchpaper-ai-mark.svg"),
    size="large",
)

st.markdown(
    """
    <style>
    [data-testid="stBottom"] {
        border-top: 1px solid rgba(128, 128, 128, 0.25);
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
    }

    /* ---- Pulsing / breathing glow bar (single accent color, no sweep) */
    .glow-bar-wrap { padding: 4px 0 10px 0; }
    .glow-bar {
        height: 4px;
        width: 100%;
        border-radius: 999px;
        background: var(--primary-color, #2563EB);
        animation: glow-breathe 1.6s ease-in-out infinite;
    }
    @keyframes glow-breathe {
        0%, 100%   { opacity: 0.28; box-shadow: 0 0 2px 0px var(--primary-color, #2563EB); }
        50%        { opacity: 1;    box-shadow: 0 0 14px 4px var(--primary-color, #2563EB); }
    }

    /* Blinking typing cursor used during progressive answer reveal */
    .type-cursor { animation: blink 0.9s steps(1) infinite; opacity: 0.6; }
    @keyframes blink { 50% { opacity: 0; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, citations, query_type}
if "preview" not in st.session_state:
    st.session_state.preview = None  # {source, page, snippet, paper_title}

library.ensure_library()


def set_preview(citation: dict):
    st.session_state.preview = citation


# PDF page rendering helper 
@st.cache_data(show_spinner=False)
def render_pdf_page(source_path: str, page_number: int, snippet: str = "") -> bytes:
    """Render a single PDF page to a PNG, highlighting the cited snippet
    (a "hover-to-see" style preview: the exact passage is boxed in yellow)."""
    doc = fitz.open(source_path)
    page = doc[page_number - 1]

    if snippet:
        needle = " ".join(snippet.split())[:80]
        for rect in page.search_for(needle):
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 0.85, 0))
            highlight.update()

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def render_citation_buttons(msg):
    if not msg.get("citations"):
        return
    st.caption("Sources:")
    cols = st.columns(min(len(msg["citations"]), 4) or 1)
    for i, cit in enumerate(msg["citations"]):
        label = f"[{cit['paper_title']}, p.{cit['page']}]"
        with cols[i % len(cols)]:
            if st.button(label, key=f"cite_{id(msg)}_{i}"):
                set_preview(cit)


def read_evaluation_report():
    report_path = config.EVAL_DIR / "results.json"
    if not report_path.exists():
        return None, None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not read the evaluation report: {exc}"
    if not isinstance(report, dict):
        return None, "The evaluation report must contain a JSON object."
    return report, None


def render_dashboard(collection_name, collection_documents, index_state, collection_count):
    question_count = sum(
        message.get("role") == "user"
        for message in st.session_state.messages
    )
    indexed_count = sum(
        document.get("status") == "indexed"
        for document in collection_documents
    )

    st.title("Research dashboard")
    st.caption(f"Library overview for **{collection_name}**")
    with st.grid(4, min_column_width=160, gap="small"):
        st.metric("Papers", len(collection_documents), border=True)
        st.metric(
            "Indexed papers",
            f"{indexed_count}/{len(collection_documents)}",
            border=True,
        )
        st.metric("Collections", collection_count, border=True)
        st.metric("Questions this session", question_count, border=True)

    if index_state["status"] not in {"indexed", "legacy"} or index_state["is_stale"]:
        st.warning("The active collection needs an index build or refresh.")
    if index_state["error"]:
        st.error(index_state["error"])

    overview, activity = st.columns([1.2, 1], gap="large")
    with overview:
        with st.container(border=True):
            st.subheader("Paper indexing status")
            if collection_documents:
                status_rows = [
                    {
                        "Paper": document["filename"],
                        "Status": document["status"].title(),
                        "Details": document["status_message"] or "",
                    }
                    for document in collection_documents
                ]
                st.dataframe(
                    status_rows,
                    column_config={
                        "Paper": st.column_config.TextColumn("Paper", width="medium"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                        "Details": st.column_config.TextColumn("Details", width="small"),
                    },
                    hide_index=True,
                    height=min(320, max(120, 36 * len(status_rows) + 38)),
                    alt="PDF indexing status for the selected collection",
                )
            else:
                st.info("No papers in this collection yet.")

            if config.SKIPPED_LOG_PATH.exists():
                skipped = json.loads(config.SKIPPED_LOG_PATH.read_text(encoding="utf-8"))
                if skipped:
                    with st.expander(f"Skipped or unreadable PDFs ({len(skipped)})"):
                        st.dataframe(
                            [
                                {"PDF": item["file"], "Reason": item["reason"]}
                                for item in skipped
                            ],
                            hide_index=True,
                            alt="PDFs skipped during ingestion and their reasons",
                        )

    with activity:
        with st.container(border=True):
            st.subheader("Recent questions")
            recent_questions = [
                message["content"]
                for message in st.session_state.messages
                if message.get("role") == "user"
            ][-5:]
            if recent_questions:
                for question_text in reversed(recent_questions):
                    st.markdown(f"> {question_text}")
            else:
                st.info("Questions you ask in this session will appear here.")

    st.subheader("Retrieval benchmark")
    report, report_error = read_evaluation_report()
    if report_error:
        st.error(report_error)
        return
    if report is None:
        st.info("Run `python scripts/evaluate.py --k 8` to populate retrieval metrics.")
        return

    strategies = report.get("retrieval_strategies")
    if not isinstance(strategies, dict) or not strategies:
        st.info(
            "This report uses an older format. Re-run `python scripts/evaluate.py --k 8` "
            "to compare all retrieval strategies."
        )
        return

    strategy_labels = {
        "dense_only": "Dense",
        "hybrid_only": "Hybrid",
        "hybrid_rerank": "Hybrid + rerank",
    }
    k = report.get("k", 8)
    hit_rate_key = f"hit_rate@{k}"
    metric_rows = []
    chart_rows = []
    for strategy_name, label in strategy_labels.items():
        strategy = strategies.get(strategy_name)
        if not isinstance(strategy, dict):
            continue
        metrics = strategy.get("metrics", {})
        latency = strategy.get("latency_ms", {})
        if not isinstance(metrics, dict) or not isinstance(latency, dict):
            continue
        hit_rate = metrics.get(hit_rate_key, 0.0)
        recall = metrics.get(f"recall@{k}", 0.0)
        mrr = metrics.get("mrr", 0.0)
        ndcg = metrics.get(f"ndcg@{k}", 0.0)
        metric_rows.append(
            {
                "Strategy": label,
                f"Hit rate @{k}": hit_rate,
                f"Recall @{k}": recall,
                "MRR": mrr,
                f"nDCG @{k}": ndcg,
                "Mean latency (ms)": latency.get("mean", 0.0),
                "p95 latency (ms)": latency.get("p95", 0.0),
            }
        )
        chart_rows.append(
            {
                "Strategy": label,
                "Hit rate": hit_rate,
                "MRR": mrr,
            }
        )

    if not metric_rows:
        st.warning("The evaluation report contains no recognized retrieval strategies.")
        return

    st.caption(
        f"{report.get('n_questions', 0)} questions · k={k} · "
        f"{report.get('api_usage', {}).get('total_requests', 0)} Gemini requests"
    )
    st.bar_chart(
        chart_rows,
        x="Strategy",
        y=["Hit rate", "MRR"],
        stack=False,
        height=260,
        alt="Hit rate and mean reciprocal rank by retrieval strategy",
    )
    st.dataframe(
        metric_rows,
        hide_index=True,
        alt="Retrieval quality and latency by strategy",
    )

    difficult_rows = []
    difficulty_metrics = report.get("metrics_by_difficulty", {})
    if isinstance(difficulty_metrics, dict):
        for strategy_name, values in difficulty_metrics.items():
            hard_metrics = values.get("hard") if isinstance(values, dict) else None
            if isinstance(hard_metrics, dict):
                difficult_rows.append(
                    {
                        "Strategy": strategy_labels.get(strategy_name, strategy_name),
                        f"Hit rate @{k}": hard_metrics.get(hit_rate_key, 0.0),
                        "MRR": hard_metrics.get("mrr", 0.0),
                    }
                )
    if difficult_rows:
        st.subheader("Challenging questions")
        st.dataframe(
            difficult_rows,
            hide_index=True,
            alt="Retrieval metrics for questions labeled hard",
        )

    failures = report.get("failure_cases", {})
    failure_rows = []
    if isinstance(failures, dict):
        for strategy_name, examples in failures.items():
            if not isinstance(examples, list):
                continue
            for example in examples:
                if isinstance(example, dict):
                    failure_rows.append(
                        {
                            "Strategy": strategy_labels.get(strategy_name, strategy_name),
                            "Question": example.get("question", ""),
                            "Expected papers": ", ".join(
                                example.get("expected_papers", [])
                            ),
                            "Retrieved papers": ", ".join(
                                example.get("retrieved_papers", [])
                            ),
                        }
                    )
    if failure_rows:
        with st.expander(f"Failure examples ({len(failure_rows)})"):
            st.dataframe(
                failure_rows,
                hide_index=True,
                alt="Questions where the expected paper was not retrieved",
            )


with st.sidebar:
    st.caption("Workspace")
    active_view = st.segmented_control(
        "Workspace",
        options=["Research", "Dashboard"],
        default="Research",
        required=True,
        key="active_view",
        label_visibility="collapsed",
        width="stretch",
    )

    with st.expander("Library controls", expanded=True):
        with st.form("create_collection_form"):
            new_collection_name = st.text_input("New collection name")
            create_collection_clicked = st.form_submit_button("Create collection")
        if create_collection_clicked:
            try:
                collection_id = library.create_collection(new_collection_name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["selected_collection_id"] = collection_id
                st.rerun()

        collections = library.list_collections()
        collection_names = {item["id"]: item["name"] for item in collections}
        collection_ids = list(collection_names)
        if not collection_ids:
            st.error("Create a collection before adding papers.")
            st.stop()
        if "selected_collection_id" not in st.session_state or (
            st.session_state["selected_collection_id"] not in collection_names
        ):
            st.session_state["selected_collection_id"] = collection_ids[0]
        active_collection_id = st.selectbox(
            "Collection",
            options=collection_ids,
            format_func=lambda value: collection_names[value],
            key="selected_collection_id",
        )
        if active_collection_id is None:
            st.error("Select a collection before managing papers.")
            st.stop()

        collection_documents = library.list_documents(active_collection_id)
        st.caption(f"{len(collection_documents)} papers in this collection")

        upload_key = f"pdf_uploads_{st.session_state.get('upload_generation', 0)}"
        uploaded_files = st.file_uploader(
            "Upload research papers",
            type=["pdf"],
            accept_multiple_files=True,
            key=upload_key,
        )
        if st.button("Upload and index", disabled=not uploaded_files, width="stretch"):
            uploaded_count = 0
            for uploaded_file in uploaded_files:
                try:
                    library.add_pdf(
                        active_collection_id,
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )
                except library.DuplicateDocumentError as exc:
                    st.warning(str(exc))
                except ValueError as exc:
                    st.error(f"{uploaded_file.name}: {exc}")
                else:
                    uploaded_count += 1
            if uploaded_count:
                try:
                    with st.spinner("Indexing the collection..."):
                        chunk_count = library.rebuild_library_index()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Indexing failed: {exc}")
                else:
                    st.success(
                        f"Added {uploaded_count} PDF(s) and indexed {chunk_count} chunks."
                    )
                    st.session_state["upload_generation"] = (
                        st.session_state.get("upload_generation", 0) + 1
                    )
                    st.rerun()

        if collection_documents:
            removal_ids = [document["id"] for document in collection_documents]
            removal_labels = {
                document["id"]: document["filename"] for document in collection_documents
            }
            removal_id = st.selectbox(
                "PDF to remove",
                options=removal_ids,
                format_func=lambda value: removal_labels[value],
                key="document_to_remove",
            )
            confirm_removal = st.checkbox("Confirm PDF removal")
            if removal_id is not None and st.button(
                "Remove PDF",
                disabled=not confirm_removal,
                icon=":material/delete:",
                width="stretch",
            ):
                try:
                    removed_filename = library.remove_document(removal_id)
                    with st.spinner("Rebuilding the index..."):
                        chunk_count = library.rebuild_library_index()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not remove and reindex the PDF: {exc}")
                else:
                    st.success(f"Removed {removed_filename}; indexed {chunk_count} chunks.")
                    st.rerun()

    index_state = library.get_index_state()
    status_colors: dict[str, StatusColor] = {
        "indexed": "green",
        "legacy": "blue",
        "pending": "orange",
        "stale": "orange",
        "indexing": "blue",
        "failed": "red",
        "empty": "gray",
    }
    st.badge(
        f"Index {index_state['status'].title()}",
        color=status_colors.get(index_state["status"], "gray"),
    )
    if index_state["error"]:
        st.error(index_state["error"])
    if index_state["status"] not in {"indexed", "legacy"} or index_state["is_stale"]:
        st.caption("Build or refresh the index before asking questions.")
        if st.button(
            "Build / refresh index",
            icon=":material/refresh:",
            width="stretch",
        ):
            try:
                with st.spinner("Building the hybrid index..."):
                    chunk_count = library.rebuild_library_index()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Indexing failed: {exc}")
            else:
                st.success(f"Indexed {chunk_count} chunks.")
                st.rerun()

    st.caption(
        "Add or remove a paper to rebuild the hybrid index. "
        "You can also run `python scripts/build_index.py`."
    )
    if st.button(
        "Clear chat",
        icon=":material/delete_sweep:",
        width="stretch",
    ):
        st.session_state.messages = []
        st.session_state.preview = None
        st.rerun()


    collection_name = collection_names[active_collection_id]

st.title("ResearchPaper AI")
st.caption(
    "Your AI-powered research companion. "
    "Explore, understand, and compare academic literature "
    "using intelligent retrieval and grounded AI responses."
)

if active_view == "Dashboard":
    render_dashboard(
        collection_name,
        collection_documents,
        index_state,
        len(collections),
    )
else:
    if not config.GOOGLE_API_KEY:
        st.warning(
            "GOOGLE_API_KEY is not set. Add it to your `.env` file before asking questions.",
            icon=":material/key_off:",
        )

    can_answer = (
        index_state["status"] in {"indexed", "legacy"} and not index_state["is_stale"]
    )
    if not can_answer:
        if not library.list_documents():
            st.info("Upload PDFs and build the index before asking questions.")
        else:
            st.warning("Build or refresh the collection index before asking questions.")

    paper_titles = indexing.list_indexed_papers(active_collection_id)
    research_task: AssistantTask = st.selectbox(
        "Research task",
        options=["Ask", "Summary", "Contributions", "Limitations", "Compare methodologies"],
    )
    selected_papers: list[str] = []
    if research_task != "Ask":
        selected_papers = st.multiselect(
            "Target papers (leave empty to use the whole collection)",
            options=paper_titles,
        )
    if research_task == "Compare methodologies" and len(selected_papers) < 2:
        st.info("Select at least two papers to compare their methodologies.")

    left, right = st.columns([3, 2], gap="large")
    question = st.chat_input(
        "Ask about your papers (e.g. 'What dataset did the RAPTOR paper use?')"
    )

    with left:
        chat_container = st.container(height=620)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    render_citation_buttons(msg)

        if question and not can_answer:
            st.warning("Build or refresh the index before asking a question.")
        elif question and research_task == "Compare methodologies" and len(selected_papers) < 2:
            st.warning("Select at least two papers before comparing methodologies.")
        elif question:
            submitted_question = question
            st.session_state.messages.append({"role": "user", "content": submitted_question})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(submitted_question)

                with st.chat_message("assistant"):
                    result_holder = {}

                    def _worker():
                        try:
                            result_holder["result"] = answer_question(
                                submitted_question,
                                collection_id=active_collection_id,
                                paper_titles=selected_papers or None,
                                task=research_task,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logging.exception("Question answering failed")
                            result_holder["error"] = exc

                    worker_thread = threading.Thread(target=_worker, daemon=True)
                    worker_thread.start()

                    stages = [
                        "Understanding your question…",
                        "Routing to the best retrieval strategy…",
                        "Searching your papers (hybrid search)…",
                        "Re-ranking the most relevant passages…",
                        "Drafting a grounded answer…",
                    ]

                    with st.status("🧠 Thinking…", expanded=True) as status:
                        glow_slot = st.empty()
                        glow_slot.markdown(
                            '<div class="glow-bar-wrap"><div class="glow-bar"></div></div>',
                            unsafe_allow_html=True,
                        )
                        i = 0
                        while worker_thread.is_alive():
                            status.update(label=f"🧠 {stages[i % len(stages)]}")
                            i += 1
                            time.sleep(0.85)
                        worker_thread.join()
                        glow_slot.empty()

                        if "error" in result_holder:
                            status.update(
                                label="⚠️ Something went wrong",
                                state="error",
                                expanded=False,
                            )
                        else:
                            status.update(
                                label="✅ Answer ready",
                                state="complete",
                                expanded=False,
                            )

                    answer_slot = st.empty()
                    if "error" in result_holder:
                        full_answer = (
                            f"⚠️ {format_answering_error(result_holder['error'])}"
                        )
                        answer_slot.markdown(full_answer)
                        citations, query_type = [], None
                    else:
                        result = result_holder["result"]
                        full_answer = result.answer
                        citations = result.citations
                        query_type = getattr(result, "query_type", None)

                        displayed = ""
                        words = full_answer.split(" ")
                        for idx in range(0, len(words), 3):
                            displayed += (
                                (" " if displayed else "")
                                + " ".join(words[idx : idx + 3])
                            )
                            answer_slot.markdown(
                                displayed + ' <span class="type-cursor">▌</span>',
                                unsafe_allow_html=True,
                            )
                            time.sleep(0.025)
                        answer_slot.markdown(displayed)

                    new_msg = {
                        "role": "assistant",
                        "content": full_answer,
                        "citations": citations,
                        "query_type": query_type,
                    }
                    render_citation_buttons(new_msg)

            st.session_state.messages.append(new_msg)
            if citations:
                set_preview(citations[0])
            st.rerun()

    with right:
        st.subheader("Research paper preview")
        preview_container = st.container(height=620)
        with preview_container:
            preview = st.session_state.preview
            if preview is None:
                st.info("Select a citation from an answer to preview its source page.")
            else:
                st.markdown(f"**{preview['paper_title']}** — page {preview['page']}")
                source_path = preview.get("source")
                try:
                    img_bytes = render_pdf_page(
                        source_path,
                        int(preview["page"]),
                        preview.get("snippet", ""),
                    )
                    st.image(
                        Image.open(io.BytesIO(img_bytes)),
                        width="stretch",
                        alt=f"Page {preview['page']} of {preview['paper_title']}",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not render page: {exc}")
                with st.expander("Cited snippet"):
                    st.write(preview.get("snippet", ""))
