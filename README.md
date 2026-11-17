# Academic Literature RAG Assistant

A domain-specific Retrieval-Augmented Generation (RAG) system for research/thesis
"related work" folders. Drop all the reference PDFs into a folder, ask questions in a chat UI, and
get answers with **clickable citations** that open the exact cited page in a live PDF
preview panel — no more manually re-reading or re-uploading PDFs to a chatbot.

Built with local components and a configurable Gemini API:

| Component        | Choice                                   
|-------------------|-------------------------------------------|
| LLM               | Google **Gemini** (configured with `GEMINI_MODEL`) |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2`  | 
| Vector DB         | **ChromaDB** (Free, local)                | 
| Keyword search    | `rank_bm25`  (Free, local)                | 
| Re-ranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2` (Free, replaces paid Cohere Rerank) | 
| UI                | **Streamlit**                             | 
| PDF parsing/render| **PyMuPDF**                               |

I built this **project-based learning** for the concepts in
LangChain's [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch) notebook series (indexing, hybrid retrieval,
query routing, decomposition, re-ranking). See [Concepts & where they live](#concepts--where-they-live-in-this-repo)
below to map each notebook idea to the actual production code.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Project structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Document management](#document-management)
7. [Research assistant](#research-assistant)
8. [Research dashboard](#research-dashboard)
9. [RAG evaluation](#rag-evaluation)
10. [Concepts & where they live in this repo](#concepts--where-they-live-in-this-repo)
11. [License](#license)

---

## How it works

```
                 ┌──────────────────────────────────────────┐
   PDFs  ──────▶ │ 1. Ingestion (src/ingestion.py)           │
 (collections)   │    - extract text per page (PyMuPDF)      │
                 │    - flag & skip scanned/no-text PDFs     │
                 │    - record paper, section, page, offsets │
                 │    - chunk with collection metadata       │
                 └───────────────────┬────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │ 2. Indexing (src/indexing.py)             │
                 │    - versioned Chroma vector + BM25 indexes│
                 │    - atomic switch after successful build │
                 └───────────────────┬────────────────────────┘
                                     ▼
 Question ─────▶ ┌──────────────────────────────────────────┐
                 │ 3. Router (src/router.py)                  │
                 │    Local rules route facts and summaries;  │
                 │    Gemini decomposes comparisons           │
                 └───────────────────┬────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │ 4. Retrieval (src/retrieval.py)            │
                 │    FACT       → hybrid search, top 20      │
                 │                 candidates → rerank → top 8│
                 │    SUMMARY    → wide per-paper pool → rerank│
                 │    COMPARISON → retrieve per sub-question,  │
                 │                 merge, rerank                │
                 └───────────────────┬────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │ 5. Generation (src/generation.py)          │
                 │    Gemini answers ONLY from retrieved       │
                 │    chunks, citing [Paper Title, p. X]       │
                 └───────────────────┬────────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │ 6. Streamlit UI (app.py)                   │
                 │    collections, document manager, task modes│
                 │    citations + source-page preview         │
                 └──────────────────────────────────────────┘
```

## Project structure

```
academic-rag/
├── app.py                  # Streamlit UI and document controls
├── .streamlit/
│   └── config.toml         # Custom light and dark themes
├── assets/
│   ├── researchpaper-ai-logo.svg
│   └── researchpaper-ai-mark.svg
├── config.py               
├── requirements.txt
├── .env.example
├── LICENSE                 
├── src/
│   ├── ingestion.py         # PDF text extraction, scanned-PDF filter, contextual headers, chunking
│   ├── indexing.py          # Chroma + BM25 index build/load
│   ├── retrieval.py         # Hybrid search (ensemble) + cross-encoder re-ranking
│   ├── router.py            # Query classification + decomposition (Gemini structured output)
│   ├── generation.py        # Prompting + citation parsing
│   ├── library.py           # Collections, PDF catalog, indexing status
│   └── pipeline.py          # Orchestrates router → retrieval → generation
├── scripts/
│   ├── build_index.py       # (Re)build the indexes from data/pdfs/
│   └── evaluate.py         # Retrieval and optional answer-quality metrics
├── eval/
│   └── sample_qa.json       
└── data/
    ├── pdfs/                 # PDFs storage
    └── chroma_db/            # generated vector index
```

## Installation

**Requirements:** Python 3.10–3.12.

```bash
git clone https://github.com/Misiker101/academic-rag-assistant.git

cd academic-rag-assistant

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

The first run will download the embedding model (~90 MB) and the re-ranker model
(~80 MB) from HuggingFace automatically (this needs internet access once, after
which they're cached locally).

## Configuration

1. Get a **free** Gemini API key: https://aistudio.google.com/app/apikey
2. Copy the env template and fill it in:

   ```bash
   cp .env.example .env
   ```

   ```env
   GOOGLE_API_KEY=your_actual_key_here
   ```

3. (Optional) Tune retrieval/chunking behavior in `.env` or `config.py` — chunk size,
   overlap, `TOP_K_CANDIDATES` (default 20, per your "high precision" requirement),
   `TOP_K_FINAL` (chunks actually sent to the LLM), hybrid search weighting, etc.
   You can also set `GEMINI_MODEL` to a model available to your Google project.

Gemini request quotas depend on the project, model, and billing plan. The app does not
automatically retry quota-exhausted requests. For an exhausted quota, wait for the
provider's reset window or configure a model/project with available quota.

## Usage

1. **Add your PDFs.** Upload PDFs through the app's **Library controls** sidebar,
   or copy them into `data/pdfs/`.

2. **Build the index** (run again any time you add/remove PDFs):

   ```bash
   python scripts/build_index.py
   ```

   Skipped files (for example, scanned PDFs with no extractable text) are logged to
   `data/skipped_pdfs.json`.

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   Open the URL Streamlit prints (defaults to `http://localhost:8501`).

4. Choose **Research** or **Dashboard** from the sidebar. In **Research**, choose
   a research task and ask questions, e.g.:
   - *"Summarize the main contribution of the XYZ paper."* → routed to **SUMMARY**
   - *"What accuracy did the ViT paper report on the trivia QA benchmark?"* → **FACT**
   - *"How do the re-ranking approaches in these papers differ from the query
     decomposition approaches?"* → **COMPARISON** (decomposed into sub-questions,
     retrieved per aspect, then synthesized)

   Click any `[Paper Title, p. X]` citation button under an answer to load that exact
   page in the right-hand preview panel, with the cited passage highlighted.

## Document management

The sidebar lets you create collections, upload multiple PDFs, remove a selected
PDF (with confirmation), and see per-file indexing status. Adding or removing a
PDF automatically rebuilds the hybrid index. Existing PDFs directly under
`data/pdfs/` are registered in the **General** collection; uploaded files are
stored in collection-specific subfolders. The library catalog is local SQLite
(`data/library.sqlite3`, ignored by Git).

Each rebuild creates a new Chroma, BM25, and document index version, then switches
the active manifest only after the new version is complete. If a rebuild fails,
the previous index is retained and the sidebar shows the error.

## Research assistant

Select one of the research tasks above the chat:

- **Ask**: route a question to fact, summary, comparison, or general retrieval.
- **Summary**: synthesize the paper's problem, approach, findings, and conclusion.
- **Contributions**: extract contributions stated in the selected paper passages.
- **Limitations**: identify stated limitations and label any inference.
- **Compare methodologies**: select at least two papers for a grounded comparison
  of methods, data, evaluation, and trade-offs.

Task responses use retrieved passages and retain clickable paper/page citations.
For paper-specific tasks, select target papers; leaving the selection empty uses
the current collection.

## Research dashboard

The sidebar's **Dashboard** workspace summarizes the selected collection with
paper and indexing-status counts, recent questions, and any retrieval benchmark
report. The benchmark view compares dense, hybrid, and hybrid-plus-reranking
quality and latency, including difficult questions and example failures. Run
`python scripts/evaluate.py --k 8` to generate the report shown there.

The app includes a custom ResearchPaper AI wordmark and icon, plus coordinated
light and dark Streamlit themes configured in `.streamlit/config.toml`.

## RAG evaluation

The evaluation set is `eval/sample_qa.json`. Each item should include a question
and either `expected_paper` or `expected_papers`. The set includes `category` and
`difficulty` labels for a small challenging subset. Retrieval evaluation compares
dense-only, hybrid-only, and hybrid-plus-reranking, reporting Hit Rate@k, Recall@k,
MRR, nDCG@k, mean/p50/p95 retrieval latency, metrics by difficulty, and up to three
failure examples per strategy:

```bash
python scripts/evaluate.py --k 8
```

Retrieval-only evaluation makes no Gemini calls. Add `--answer-quality` to also
generate answers, score faithfulness and answer relevance with a Gemini judge, and
compute exact paper/page citation precision. This mode requires `GOOGLE_API_KEY`
and reports generation/judge request counts and answer-quality latency; token usage
is not currently collected. Use `--limit` for a smaller run and `--failure-limit`
to change the number of examples shown per strategy. Reports are written to
`eval/results.json`.

For regular “Ask” questions, obvious fact and summary routing uses local rules;
Gemini is reserved for comparison-query decomposition and answer generation.
Successful answers are held in a bounded in-memory cache keyed by the active index
version/revision, collection, selected papers, model, task, and question. A newly
activated index invalidates prior entries.

## Concepts & where they live in this repo

| Notebook concept | Notebook part | Where it's used here |
|---|---|---|
| Indexing (load → split → embed → vectorstore) | Part 2 | `src/ingestion.py`, `src/indexing.py` |
| Retrieval & generation basics | Parts 1–4 | `src/retrieval.py`, `src/generation.py` |
| Query routing (function calling / structured output) | Part 10 | `src/router.py` |
| Query decomposition (sub-questions, answer then synthesize) | Part 7 | `src/router.py` (`sub_questions`) + `retrieval.retrieve_multi` |
| Query construction (metadata filtering) | Part 11 | `retrieval.filter_by_paper` (paper-title hint filtering) |
| Re-ranking | Part 15 | `retrieval.rerank` (local cross-encoder instead of paid Cohere) |
| RAG-Fusion (reciprocal rank fusion of multiple retrieved lists) | Part 6 | `retrieval.build_hybrid_retriever` (`EnsembleRetriever` fuses dense + BM25 rankings) |
| Metadata filtering | Part 11 | `retrieval.build_hybrid_retriever` filters by collection and paper |
| Multi-representation / parent-document style broad retrieval | Part 12 | `retrieval.retrieve_for_summary` (wide per-paper pool for summarization) |

Chunking preserves page boundaries and records chunk offsets plus paper, section,
collection, and source metadata. Retrieval uses weighted dense/BM25 reciprocal-rank
fusion, strict metadata filters, and a local cross-encoder reranker.

## License

MIT — see [`LICENSE`](LICENSE).