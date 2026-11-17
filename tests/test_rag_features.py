import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz
from langchain_core.documents import Document

import config
from scripts.evaluate import _retrieval_metrics
from src import generation, indexing, ingestion, library, retrieval


class RetrievalFeatureTests(unittest.TestCase):
    def test_bm25_tokenizer_keeps_compounds_and_searchable_parts(self):
        self.assertEqual(
            indexing.tokenize("cross-encoder top-1"),
            ["cross-encoder", "cross", "encoder", "top-1", "top", "1"],
        )

    def test_collection_and_paper_filters_are_combined(self):
        self.assertEqual(
            retrieval._chroma_filter("collection-a", ["Paper A", "Paper B"]),
            {
                "$and": [
                    {"collection_id": "collection-a"},
                    {"paper_title": {"$in": ["Paper A", "Paper B"]}},
                ]
            },
        )
        self.assertFalse(
            retrieval._matches_filters(
                Document(
                    page_content="text",
                    metadata={"collection_id": "collection-a", "paper_title": "Paper A"},
                ),
                "collection-b",
                ["Paper A"],
            )
        )

    def test_missing_paper_hint_does_not_broaden_results(self):
        docs = [
            Document(page_content="text", metadata={"paper_title": "Paper A"}),
        ]
        self.assertEqual(retrieval.filter_by_paper(docs, "Paper B"), [])

    def test_citation_correctness_requires_exact_source_page(self):
        docs = [
            Document(
                page_content="evidence",
                metadata={"paper_title": "Paper A", "page": 3},
            )
        ]
        result = generation.citation_correctness(
            "Supported [Paper A, p. 3], unsupported [Paper B, p. 4].",
            docs,
        )
        self.assertEqual(result, {"correct": 1, "total": 2, "precision": 0.5})

    def test_retrieval_metrics_cover_recall_rank_and_ndcg(self):
        docs = [
            Document(page_content="first", metadata={"paper_title": "Paper B"}),
            Document(page_content="second", metadata={"paper_title": "Unrelated"}),
            Document(page_content="third", metadata={"paper_title": "Paper A"}),
        ]
        metrics = _retrieval_metrics([(docs, ["Paper A", "Paper B"])], k=3)
        self.assertEqual(metrics["hit_rate@3"], 1.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertLess(metrics["ndcg@3"], 1.0)

    def test_index_build_publishes_a_complete_version(self):
        document = Document(
            page_content="Dense and keyword retrieval",
            metadata={"collection_id": "general", "paper_title": "Paper A"},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(config, "INDEX_VERSIONS_DIR", root / "versions"),
                patch.object(config, "INDEX_MANIFEST_PATH", root / "active.json"),
                patch.object(indexing, "build_vectorstore"),
            ):
                indexing.build_all_indexes([document], library_revision=7)
                self.assertEqual(indexing.active_index_revision(), 7)
                self.assertEqual(indexing.load_docstore(), [document])
                self.assertEqual(indexing.load_bm25_index()["docs"], [document])

    def test_empty_index_does_not_fall_back_to_legacy_storage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(config, "INDEX_VERSIONS_DIR", root / "versions"),
                patch.object(config, "INDEX_MANIFEST_PATH", root / "active.json"),
            ):
                indexing.write_empty_index(library_revision=2)
                self.assertTrue(indexing.is_empty_index())
                self.assertEqual(indexing.load_docstore(), [])


class LibraryFeatureTests(unittest.TestCase):
    def test_pdf_upload_and_removal_updates_collection_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(config, "PDF_DIR", root / "pdfs"),
                patch.object(config, "LIBRARY_DB_PATH", root / "library.sqlite3"),
                patch.object(config, "BM25_INDEX_PATH", root / "bm25.pkl"),
                patch.object(config, "CHROMA_DIR", root / "chroma_db"),
            ):
                collection_id = library.create_collection("Vision")
                uploaded = library.add_pdf(
                    collection_id,
                    r"..\unsafe\paper.pdf",
                    b"%PDF-1.7 test content",
                )
                self.assertEqual(uploaded["filename"], "paper.pdf")
                self.assertTrue(uploaded["path"].is_file())
                self.assertEqual(library.list_documents(collection_id)[0]["status"], "pending")
                with self.assertRaises(library.DuplicateDocumentError):
                    library.add_pdf(collection_id, "copy.pdf", b"%PDF-1.7 test content")
                self.assertEqual(library.remove_document(uploaded["id"]), "paper.pdf")
                self.assertFalse(uploaded["path"].exists())
                self.assertEqual(library.list_documents(collection_id), [])

    def test_ingestion_records_collection_and_chunk_offsets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_path = root / "collection-a" / "paper.pdf"
            pdf_path.parent.mkdir()
            with fitz.open() as pdf:
                page = pdf.new_page()
                page.insert_text(
                    (72, 72),
                    "This research paper contains enough searchable text for ingestion and "
                    "retrieval metadata validation.",
                )
                pdf.save(pdf_path)
            with patch.object(config, "SKIPPED_LOG_PATH", root / "skipped.json"):
                chunks = ingestion.load_and_chunk_pdfs(root)
            self.assertTrue(chunks)
            self.assertEqual(chunks[0].metadata["collection_id"], "collection-a")
            self.assertGreaterEqual(chunks[0].metadata["char_start_index"], 0)


if __name__ == "__main__":
    unittest.main()
