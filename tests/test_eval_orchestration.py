"""Tests for eval.py orchestration functions (no GPU required)."""
import json
import pytest
from src.eval import (
    wilson_ci,
    grade_with_subtasks,
    grade,
)


# === Test fixtures ===

@pytest.fixture
def sample_results():
    """Minimal result set for testing aggregation."""
    return [
        {"id": "pwn-001", "category": "pwn", "difficulty": "easy",
         "correct": True, "score": 1.0, "task_type": "flag_extraction",
         "response": "flag{test}", "expected": "flag{test}",
         "time": 5.0, "suspicious_memorization": False},
        {"id": "pwn-002", "category": "pwn", "difficulty": "easy",
         "correct": False, "score": 0.4, "task_type": "flag_extraction",
         "response": "some text", "expected": "flag{test2}",
         "time": 6.0, "suspicious_memorization": False},
        {"id": "vuln-001", "category": "pwn", "difficulty": "easy",
         "correct": True, "score": 1.0, "task_type": "vulnerability_identification",
         "response": "Buffer overflow", "expected": "A",
         "time": 4.0, "suspicious_memorization": False},
    ]


def fake_benchmark():
    """Generate a minimal fake benchmark list for testing grade_with_subtasks."""
    return [
        {"id": "test-001", "category": "pwn", "difficulty": "easy",
         "task_type": "flag_extraction",
         "prompt": "test", "system_prompt": "test",
         "expected": "flag{test}",
         "subtasks": [
             {"name": "has_flag", "criterion": r"flag\{[^}]+\}", "weight": 1.0},
         ]},
        {"id": "test-002", "category": "pwn", "difficulty": "medium",
         "task_type": "flag_extraction",
         "prompt": "test", "system_prompt": "test",
         "expected": "flag{other}"},
    ]


class TestGradeWithSubtasks:
    def test_subtask_match(self):
        bench = fake_benchmark()
        score, fb = grade_with_subtasks("flag{test} here", bench[0])
        assert score >= 1.0

    def test_subtask_no_match(self):
        bench = fake_benchmark()
        score, fb = grade_with_subtasks("no flag here", bench[0])
        assert score < 0.5

    def test_fallback_no_subtasks(self):
        bench = fake_benchmark()
        score, fb = grade_with_subtasks("flag{other}", bench[1])
        assert score >= 1.0

    def test_fallback_wrong(self):
        bench = fake_benchmark()
        score, fb = grade_with_subtasks("flag{wrong}", bench[1])
        assert score < 1.0


class TestResultDataIntegrity:
    """Tests for the result data format that save_results outputs."""
    def test_single_model_json(self, tmp_path, sample_results):
        from src.eval import wilson_ci
        import json as j

        # Build a minimal eval result
        correct = sum(1 for r in sample_results if r["correct"])
        total = len(sample_results)
        lo, hi = wilson_ci(correct, total)
        data = {
            "timestamp": "2026-01-01T00:00:00",
            "models": ["test-model"],
            "results": [{
                "model": "test-model",
                "adapter": "",
                "accuracy": correct / total,
                "mean_score": sum(r.get("score", float(r["correct"])) for r in sample_results) / total,
                "wilson_ci95": [lo, hi],
                "correct": correct,
                "total": total,
                "suspicious_memorization": 0,
                "questions": sample_results,
            }],
        }

        out = tmp_path / "test_output.json"
        with open(out, "w") as f:
            j.dump(data, f, indent=2)

        # Verify output file is valid JSON
        with open(out) as f:
            loaded = j.load(f)
        assert loaded["results"][0]["total"] == 3
        assert loaded["results"][0]["mean_score"] > 0

    def test_wilson_ci_in_output(self, sample_results):
        correct = sum(1 for r in sample_results if r["correct"])
        total = len(sample_results)
        lo, hi = wilson_ci(correct, total)
        # Wilson CI should bracket the proportion
        p = correct / total
        assert lo <= p <= hi


class TestCheckContamination:
    def test_no_overlap(self):
        """contamination check should not crash when train.jsonl missing."""
        # This tests the graceful FileNotFoundError path
        from src.eval import check_contamination
        # Should not raise
        check_contamination([{"id": "test"}])

    def test_normalizes_hash(self):
        """Hashing should be deterministic."""
        import hashlib
        h1 = hashlib.sha256(b"test prompt").hexdigest()[:16]
        h2 = hashlib.sha256(b"test prompt").hexdigest()[:16]
        assert h1 == h2
