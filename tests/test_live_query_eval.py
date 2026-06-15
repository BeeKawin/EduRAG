import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from chat.api import app
from evaluation.judge import JudgeResult
from evaluation import live_query_eval


class FakeResponse:
    def __init__(self, content):
        self.content = content


class TestLiveQueryEval(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(__file__).resolve().parent / ".tmp_live_query_eval"
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for p in sorted(self.tmp_root.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass

    def test_baseline_generation_does_not_receive_retrieved_context(self):
        fake_llm = Mock()
        fake_llm.invoke.return_value = FakeResponse("baseline answer")

        with patch.object(live_query_eval, "_get_baseline_llm", return_value=fake_llm):
            answer = live_query_eval.generate_baseline_answer(
                "What is force?",
                preferred_answer_type="concept-focused",
                language="EN",
            )

        prompt = fake_llm.invoke.call_args.args[0]
        self.assertEqual(answer, "baseline answer")
        self.assertNotIn("Retrieved curriculum context", prompt)
        self.assertNotIn("secret context", prompt)
        self.assertIn("Target answer type: concept-focused", prompt)

    def test_parse_live_judge_response(self):
        raw = json.dumps(
            {
                "rag_scores": {
                    "correctness": 5,
                    "groundedness": 5,
                    "completeness": 4,
                    "clarity": 4,
                    "type_alignment": 5,
                    "detected_answer_type": "homework-help",
                    "rationale": "Grounded and useful.",
                },
                "baseline_scores": {
                    "correctness": 3,
                    "groundedness": 2,
                    "completeness": 3,
                    "clarity": 4,
                    "type_alignment": 4,
                    "detected_answer_type": "homework-help",
                    "rationale": "Less grounded.",
                },
                "winner": "rag",
                "comparison_rationale": "RAG used the context better.",
            }
        )

        result = live_query_eval.parse_live_judge_response(raw, "homework-help")

        self.assertEqual(result.winner, "rag")
        self.assertEqual(result.rag_scores.target_answer_type, "homework-help")
        self.assertEqual(result.baseline_scores.groundedness, 2)

    def test_parse_live_judge_response_clamps_malformed_numeric_scores(self):
        raw = json.dumps(
            {
                "rag_scores": {
                    "correctness": 0,
                    "groundedness": "6",
                    "completeness": 3.4,
                    "clarity": 5,
                    "type_alignment": 2,
                    "detected_answer_type": "general",
                    "rationale": "Malformed numeric scores were repaired.",
                },
                "baseline_scores": {
                    "correctness": "0",
                    "groundedness": 1,
                    "completeness": 2,
                    "clarity": 3,
                    "type_alignment": 4,
                    "detected_answer_type": "general",
                    "rationale": "Also repaired.",
                },
                "winner": "tie",
                "comparison_rationale": "Both are usable.",
            }
        )

        result = live_query_eval.parse_live_judge_response(raw, "general")

        self.assertEqual(result.rag_scores.correctness, 1)
        self.assertEqual(result.rag_scores.groundedness, 5)
        self.assertEqual(result.rag_scores.completeness, 3)
        self.assertEqual(result.baseline_scores.correctness, 1)

    def test_judge_live_answers_falls_back_when_output_is_not_json(self):
        fake_llm = Mock()
        fake_llm.invoke.return_value = FakeResponse("I cannot provide JSON for this request.")

        with patch.object(live_query_eval, "_get_judge_llm", return_value=fake_llm):
            result = live_query_eval.judge_live_answers(
                question="What is force?",
                context="Force is a push or pull.",
                rag_answer="Force is a push or pull.",
                baseline_answer="Force is an interaction.",
                preferred_answer_type="quick-answer",
                language="EN",
            )

        self.assertEqual(result.winner, "tie")
        self.assertEqual(result.rag_scores.correctness, 3)
        self.assertEqual(result.rag_scores.target_answer_type, "quick-answer")
        self.assertIn("malformed", result.comparison_rationale)

    def test_run_live_query_saves_jsonl_and_defaults_invalid_type(self):
        fake_chain = Mock()
        fake_chain.get_context_docs.return_value = [
            {
                "topic": "Newton's Laws",
                "section": "Second law",
                "content": "Force equals mass times acceleration.",
            }
        ]
        fake_chain.ask.return_value = "RAG answer"
        fake_scores = live_query_eval.LiveQueryScores(
            rag_scores=JudgeResult(5, 5, 5, 5, 5, "general", "general", 10, "strong"),
            baseline_scores=JudgeResult(3, 2, 3, 4, 4, "general", "general", 5, "weaker"),
            winner="rag",
            comparison_rationale="RAG is better grounded.",
        )

        with patch.object(live_query_eval, "LIVE_QUERY_DIR", self.tmp_root), \
             patch.object(live_query_eval, "get_rag_chain", return_value=fake_chain), \
             patch.object(live_query_eval, "generate_baseline_answer", return_value="Baseline answer"), \
             patch.object(live_query_eval, "judge_live_answers", return_value=fake_scores):
            result = live_query_eval.run_live_query_evaluation(
                message="What is Newton's second law?",
                subject="physics",
                grade="M4",
                preferred_answer_type="calculation",
                language="EN",
            )

        self.assertEqual(result["rag_answer"], "RAG answer")
        self.assertEqual(result["baseline_answer"], "Baseline answer")
        self.assertEqual(result["preferences"]["preferred_answer_type"], "general")
        self.assertEqual(result["winner"], "rag")

        saved_path = Path(result["saved_result_path"])
        rows = saved_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(rows), 1)
        saved = json.loads(rows[0])
        self.assertIn("rag_scores", saved)
        self.assertIn("baseline_scores", saved)

    def test_live_query_api_response_includes_both_answers(self):
        fake_result = {
            "question": "Q",
            "preferences": {"preferred_answer_type": "general"},
            "retrieved_sources": [],
            "rag_answer": "RAG",
            "baseline_answer": "BASE",
            "rag_scores": {"overall_band": 8},
            "baseline_scores": {"overall_band": 5},
            "winner": "rag",
            "comparison_rationale": "RAG wins.",
            "saved_result_path": "data/eval/live_queries/test.jsonl",
        }

        with patch("chat.api.run_live_query_evaluation", return_value=fake_result):
            client = TestClient(app)
            response = client.post("/eval/live-query", json={"message": "Q"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["rag_answer"], "RAG")
        self.assertEqual(payload["baseline_answer"], "BASE")
        self.assertEqual(payload["winner"], "rag")


if __name__ == "__main__":
    unittest.main()
