import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path(os.getenv("PDF_DIR", BASE_DIR / "data" / "pdfs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "data" / "chroma_db"))
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", BASE_DIR / "data" / "bm25_index.pkl"))
DOCSTORE_PATH = Path(os.getenv("DOCSTORE_PATH", BASE_DIR / "data" / "docstore.pkl"))
SKIPPED_LOG_PATH = Path(os.getenv("SKIPPED_LOG_PATH", BASE_DIR / "data" / "skipped_pdfs.json"))
LIBRARY_DB_PATH = Path(os.getenv("LIBRARY_DB_PATH", BASE_DIR / "data" / "library.sqlite3"))
INDEX_VERSIONS_DIR = DATA_DIR / "index_versions"
INDEX_MANIFEST_PATH = DATA_DIR / "active_index.json"
DEFAULT_COLLECTION_ID = "general"
EVAL_DIR = BASE_DIR / "eval"

for p in [PDF_DIR, CHROMA_DIR.parent]:
    p.mkdir(parents=True, exist_ok=True)


# Gemini API availability and quotas depend on the configured project and model.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Override with GEMINI_MODEL when the configured model is unavailable or out of quota.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Local, free cross-encoder re-ranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "academic_papers")


# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


# Scanned-PDF filter
# ---------------------------------------------------------------------------
# A page is considered "scanned / no extractable text" if it has fewer than
# this many extracted characters. If more than SCANNED_PAGE_RATIO of a
# document's pages fall below this, the whole document is skipped.
MIN_CHARS_PER_PAGE = int(os.getenv("MIN_CHARS_PER_PAGE", "40"))
SCANNED_PAGE_RATIO = float(os.getenv("SCANNED_PAGE_RATIO", "0.6"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# "High precision" candidate pool requested by the user
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))
# Final number of chunks handed to the LLM after re-ranking
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "8"))
# Weight given to the vector retriever inside the hybrid ensemble (0-1).
# The remainder (1 - HYBRID_VECTOR_WEIGHT) is given to BM25.
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))

# Broader pool used for whole-paper summarization queries
SUMMARY_TOP_K = int(os.getenv("SUMMARY_TOP_K", "30"))
