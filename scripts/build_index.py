"""
Run this once (and again whenever you add/remove PDFs) to (re)build the
Chroma vector index and BM25 keyword index.

Usage:
    python scripts/build_index.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import library  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Reading PDFs from %s", config.PDF_DIR)
    chunk_count = library.rebuild_library_index()
    logger.info("Indexed %d chunks. You can now run: streamlit run app.py", chunk_count)


if __name__ == "__main__":
    main()
