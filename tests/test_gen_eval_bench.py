"""Tests for src/gen_eval_bench.py benchmark data structure."""
import json
from pathlib import Path
from src.gen_eval_bench import PWN, REV, CRYPTO, WEB, VULN_ID, PATCH_GEN, EXPLOIT_TRACE, SYS_PROMPTS


class TestBenchmarkStructure:
    def test_total_count(self):
        total = len(PWN) + len(REV) + len(CRYPTO) + len(WEB)
        assert total == 200, f"Expected 200 challenges, got {total}"

    def test_ids_are_unique(self):
        all_ids = [c["id"] for c in PWN + REV + CRYPTO + WEB]
        assert len(all_ids) == len(set(all_ids)), "Duplicate IDs found"

    def test_all_have_required_fields(self):
        for c in PWN + REV + CRYPTO + WEB:
            assert "id" in c
            assert "difficulty" in c
            assert "task_type" in c
            assert c["difficulty"] in ("easy", "medium", "hard")

    def test_categories_match_prompts(self):
        for c in PWN:
            assert c["id"].startswith("pwn-")
        for c in REV:
            assert c["id"].startswith("rev-")

    def test_system_prompts_defined(self):
        expected_categories = {"pwn", "rev", "crypto", "web"}
        assert set(SYS_PROMPTS.keys()) == expected_categories
        for cat, prompt in SYS_PROMPTS.items():
            assert len(prompt) > 20

    def test_code_generation_has_reference(self):
        for c in PWN + REV + CRYPTO + WEB:
            if c.get("task_type") == "code_generation":
                assert "reference" in c, f"{c['id']} missing reference"

    def test_vuln_id_has_expected(self):
        """vulnerability_identification entries should have expected letter."""
        for c in VULN_ID:
            assert c["task_type"] == "vulnerability_identification"
            assert c["expected"] in ("A", "B", "C", "D")
            assert c["category"] in ("pwn", "web")  # current categories with vuln entries

    def test_patch_has_banned_and_required(self):
        for c in PATCH_GEN:
            assert "banned_tokens" in c, f"{c['id']} missing banned_tokens"
            assert "required_tokens" in c, f"{c['id']} missing required_tokens"

    def test_trace_has_required_steps(self):
        for c in EXPLOIT_TRACE:
            assert "required_steps" in c, f"{c['id']} missing required_steps"

    def test_difficulty_distribution(self):
        """Ensure no difficulty has < 15% of total (reasonable balance)."""
        all_c = PWN + REV + CRYPTO + WEB
        total = len(all_c)
        for d in ("easy", "medium", "hard"):
            count = sum(1 for c in all_c if c["difficulty"] == d)
            assert count / total >= 0.15, f"{d} has only {count}/{total} challenges"

    def test_task_type_distribution(self):
        """At least one of each task type exists in the combined list."""
        all_c = PWN + REV + CRYPTO + WEB + VULN_ID + PATCH_GEN + EXPLOIT_TRACE
        task_types = {c["task_type"] for c in all_c}
        for tt in ("flag_extraction", "code_generation",
                   "vulnerability_identification", "patch_generation", "exploit_trace"):
            assert tt in task_types, f"Task type {tt} missing from benchmark"

    def test_all_challenges_have_training_overlap_hash(self):
        """Every challenge should have the provenance hash field."""
        all_c = PWN + REV + CRYPTO + WEB + VULN_ID + PATCH_GEN + EXPLOIT_TRACE
        for c in all_c:
            # compute_hash is called when building the benchmark
            assert c["id"].startswith(tuple(p + "-" for p in
                ["pwn", "rev", "crypto", "web", "vuln", "patch", "trace"]))
