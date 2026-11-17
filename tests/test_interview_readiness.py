import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf as fitz
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

import config
from scripts import evaluate as evaluate_script
from src import generation, indexing, ingestion, pipeline, retrieval, router
from src.generation import RagResult
from src.router import RouteDecision


class _FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def as_retriever(self, search_kwargs):
        retriever = BM25Retriever.from_documents(
            self.documents,
            preprocess_func=indexing.tokenize,
        )
        retriever.k = search_kwargs["k"]
        return retriever


class InterviewReadinessTests(unittest.TestCase):
    def test_local_router_skips_gemini_for_fact_and_explicit_summary(self):
        paper_title = "Sample Retrieval Study"
        with patch.object(router, "get_router_llm", side_effect=AssertionError("Gemini called")):
            fact = router.route(
                "Which optimizer did Sample Retrieval Study use?",
                paper_titles=[paper_title],
            )
            summary = router.route(
                "Summarize Sample Retrieval Study.",
                paper_titles=[paper_title],
            )

        self.assertEqual(fact.query_type, "FACT")
        self.assertEqual(fact.paper_hint, paper_title)
        self.assertEqual(summary.query_type, "SUMMARY")
        self.assertEqual(summary.paper_hint, paper_title)

    def test_comparison_router_propagates_quota_errors(self):
        class FailingRouter:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _messages):
                raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

        with patch.object(router, "get_router_llm", return_value=FailingRouter()):
            with self.assertRaisesRegex(RuntimeError, "RESOURCE_EXHAUSTED"):
                router.route("Compare the methods in these two papers.")

    def test_answer_cache_is_invalidated_by_active_index_version(self):
        document = Document(
            page_content="Sample evidence",
            metadata={"paper_title": "Sample Retrieval Study", "page": 1},
        )
        version = {"value": "version-1"}
        pipeline.clear_answer_cache()
        with (
            patch.object(indexing, "active_index_version", side_effect=lambda: version["value"]),
            patch.object(indexing, "active_index_revision", return_value=4),
            patch.object(indexing, "list_indexed_papers", return_value=["Sample Retrieval Study"]),
            patch.object(
                router,
                "route",
                return_value=RouteDecision(query_type="FACT", paper_hint=None),
            ),
            patch.object(retrieval, "retrieve", return_value=[document]),
            patch.object(
                generation,
                "generate_answer",
                return_value=RagResult("An answer.", source_chunks=[document]),
            ) as generate,
        ):
            pipeline.answer_question("What is the sample finding?", collection_id="collection-a")
            pipeline.answer_question("What is the sample finding?", collection_id="collection-a")
            version["value"] = "version-2"
            pipeline.answer_question("What is the sample finding?", collection_id="collection-a")

        self.assertEqual(generate.call_count, 2)
        pipeline.clear_answer_cache()

    def test_api_errors_have_actionable_messages_without_retry_promises(self):
        quota_message = pipeline.format_answering_error(
            RuntimeError("429 RESOURCE_EXHAUSTED: daily quota exceeded")
        )
        timeout_message = pipeline.format_answering_error(TimeoutError("request timed out"))

        self.assertIn("quota", quota_message.lower())
        self.assertIn("not retried", quota_message)
        self.assertIn("timed out", timeout_message.lower())
        self.assertIn("not retried automatically", timeout_message)

    def test_all_gemini_clients_disable_automatic_retries(self):
        original_router = router._llm
        original_generation = generation._llm
        try:
            router._llm = None
            with patch.object(router, "ChatGoogleGenerativeAI") as router_factory:
                router.get_router_llm()
            self.assertEqual(router_factory.call_args.kwargs["retries"], 0)

            generation._llm = None
            with patch.object(generation, "ChatGoogleGenerativeAI") as answer_factory:
                generation.get_llm()
            self.assertEqual(answer_factory.call_args.kwargs["retries"], 0)

            class Judge:
                def invoke(self, _messages):
                    return generation.AnswerQualityScore(
                        faithfulness=5,
                        answer_relevance=5,
                        explanation="Supported.",
                    )

            class JudgeModel:
                def with_structured_output(self, _schema):
                    return Judge()

            with (
                patch.object(config, "GOOGLE_API_KEY", "test-key"),
                patch.object(
                    generation,
                    "ChatGoogleGenerativeAI",
                    return_value=JudgeModel(),
                ) as judge_factory,
            ):
                generation.evaluate_answer_quality("Question?", "Answer.", [])
            self.assertEqual(judge_factory.call_args.kwargs["retries"], 0)
        finally:
            router._llm = original_router
            generation._llm = original_generation

    def test_empty_generation_returns_abstention_without_calling_gemini(self):
        with patch.object(generation, "get_llm", side_effect=AssertionError("Gemini called")):
            result = generation.generate_answer("Question with no evidence", [])

        self.assertIn("couldn't find relevant passages", result.answer)
        self.assertEqual(result.citations, [])

    def test_duplicate_chunks_do_not_inflate_ndcg(self):
        documents = [
            Document(page_content="first chunk", metadata={"paper_title": "Expected Study"}),
            Document(page_content="duplicate chunk", metadata={"paper_title": "Expected Study"}),
        ]
        metrics = evaluate_script._retrieval_metrics(
            [(documents, ["Expected Study"])],
            k=2,
        )

        self.assertEqual(metrics["ndcg@2"], 1.0)

    def test_pdf_ingestion_retrieval_and_citation_resolve_to_a_real_page(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_path = root / "sample.pdf"
            with fitz.open() as pdf:
                page = pdf.new_page()
                page.insert_text(
                    (72, 72),
                    "The Sample Retrieval Study uses the Adam optimizer to train its model. "
                    "The optimizer choice is reported in the training setup.",
                )
                pdf.set_metadata({"title": "Sample Retrieval Study"})
                pdf.save(pdf_path)

            with patch.object(config, "SKIPPED_LOG_PATH", root / "skipped.json"):
                documents = ingestion.load_and_chunk_pdfs(root)
            self.assertTrue(documents)

            with (
                patch.object(indexing, "is_empty_index", return_value=False),
                patch.object(indexing, "supports_metadata_filters", return_value=False),
                patch.object(indexing, "load_vectorstore", return_value=_FakeVectorStore(documents)),
                patch.object(indexing, "load_bm25_index", return_value={"docs": documents}),
                patch.object(
                    retrieval,
                    "rerank",
                    side_effect=lambda _query, candidates, top_k: candidates[:top_k],
                ),
            ):
                matches = retrieval.retrieve(
                    "Which optimizer was used to train the model?",
                    candidate_k=3,
                    final_k=2,
                )

            self.assertTrue(matches)
            self.assertEqual(matches[0].metadata["paper_title"], "Sample Retrieval Study")
            self.assertEqual(matches[0].metadata["page"], 1)

            class CitationModel:
                def invoke(self, _messages):
                    return SimpleNamespace(
                        content="The paper uses Adam [Sample Retrieval Study, p. 1]."
                    )

            with patch.object(generation, "get_llm", return_value=CitationModel()):
                result = generation.generate_answer(
                    "Which optimizer was used to train the model?",
                    matches,
                )

            self.assertEqual(len(result.citations), 1)
            citation = result.citations[0]
            self.assertEqual(citation["page"], 1)
            with fitz.open(citation["source"]) as cited_pdf:
                self.assertGreaterEqual(cited_pdf.page_count, citation["page"])
                cited_text = cited_pdf.load_page(citation["page"] - 1).get_text()
            self.assertIn("Adam optimizer", cited_text)

    def test_evaluator_reports_three_strategies_latency_usage_and_failures(self):
        expected = "Expected Study"
        relevant = Document(
            page_content="Relevant",
            metadata={"paper_title": expected},
        )
        unrelated = Document(
            page_content="Unrelated",
            metadata={"paper_title": "Unrelated Study"},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            qa_path = root / "qa.json"
            qa_path.write_text(
                json.dumps(
                    [
                        {
                            "question": "A difficult test question?",
                            "expected_paper": expected,
                            "difficulty": "hard",
                            "category": "paraphrase",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with (
                patch.object(config, "EVAL_DIR", root / "results"),
                patch.object(
                    evaluate_script,
                    "_dense_retrieve",
                    return_value=[unrelated],
                ),
                patch.object(
                    retrieval,
                    "retrieve_hybrid",
                    return_value=[relevant],
                ),
                patch.object(retrieval, "retrieve", return_value=[relevant]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = evaluate_script.evaluate(qa_path, k=1)

        self.assertEqual(
            set(report["retrieval_strategies"]),
            {"dense_only", "hybrid_only", "hybrid_rerank"},
        )
        self.assertGreaterEqual(
            report["retrieval_strategies"]["dense_only"]["latency_ms"]["mean"],
            0,
        )
        self.assertEqual(report["api_usage"]["total_requests"], 0)
        self.assertEqual(len(report["failure_cases"]["dense_only"]), 1)
        self.assertIn("hard", report["metrics_by_difficulty"]["hybrid_only"])

    def test_evaluator_counts_optional_generation_and_judge_requests(self):
        relevant = Document(
            page_content="Evidence",
            metadata={"paper_title": "Expected Study"},
        )
        answer = RagResult("Supported answer.", source_chunks=[relevant])
        quality = generation.AnswerQualityScore(
            faithfulness=4,
            answer_relevance=5,
            explanation="Supported by the retrieved evidence.",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            qa_path = root / "qa.json"
            qa_path.write_text(
                json.dumps(
                    [{"question": "Question?", "expected_paper": "Expected Study"}]
                ),
                encoding="utf-8",
            )
            with (
                patch.object(config, "EVAL_DIR", root / "results"),
                patch.object(evaluate_script, "_dense_retrieve", return_value=[relevant]),
                patch.object(retrieval, "retrieve_hybrid", return_value=[relevant]),
                patch.object(retrieval, "retrieve", return_value=[relevant]),
                patch.object(generation, "generate_answer", return_value=answer),
                patch.object(generation, "evaluate_answer_quality", return_value=quality),
                patch.object(
                    generation,
                    "citation_correctness",
                    return_value={"correct": 1, "total": 1, "precision": 1.0},
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                report = evaluate_script.evaluate(qa_path, k=1, evaluate_answers=True)

        self.assertEqual(report["api_usage"]["generation_requests"], 1)
        self.assertEqual(report["api_usage"]["judge_requests"], 1)
        self.assertEqual(report["api_usage"]["total_requests"], 2)


if __name__ == "__main__":
    unittest.main()
