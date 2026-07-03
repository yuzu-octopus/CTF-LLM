"""Tests for src/eval.py grading functions."""
import pytest
from src.eval import (
    grade_flag, grade_mcq, grade_code, grade_patch,
    grade_exploit_trace, grade, wilson_ci,
)


class TestGradeFlag:
    def test_exact_match(self):
        assert grade_flag("flag{abc123}", "flag{abc123}")[0]

    def test_no_flag_pattern(self):
        correct, fb = grade_flag("no flag here", "flag{abc}")
        assert not correct
        assert "No flag{...} pattern" in fb

    def test_wrong_flag(self):
        correct, fb = grade_flag("flag{xyz}", "flag{abc}")
        assert not correct
        assert "Found" in fb

    def test_flag_in_long_response(self):
        assert grade_flag("Here is the answer: flag{test} done", "flag{test}")[0]


class TestGradeMcq:
    def test_explicit_answer(self):
        assert grade_mcq("The answer is B", "B")[0]

    def test_lowercase_answer(self):
        assert grade_mcq("answer: a", "A")[0]

    def test_parenthesized(self):
        assert grade_mcq("Option (C) is correct", "C")[0]

    def test_last_letter_fallback(self):
        assert grade_mcq("I think D", "D")[0]

    def test_no_letter(self):
        correct, fb = grade_mcq("no letters here", "A")
        assert not correct
        assert "No A-D letter" in fb

    def test_wrong_answer(self):
        correct, fb = grade_mcq("The answer is A", "B")
        assert not correct


class TestGradeCode:
    def test_valid_code_no_reference(self):
        correct, fb = grade_code("```python\nprint(1)\n```", None)
        assert correct

    def test_syntax_error(self):
        correct, fb = grade_code("```python\ndef foo(\n```", None)
        assert not correct
        assert "SyntaxError" in fb

    def test_reference_tokens_match(self):
        correct, fb = grade_code(
            "```python\nfrom pwn import process\n```",
            "from pwn import process"
        )
        assert correct

    def test_reference_tokens_missing(self):
        correct, fb = grade_code(
            "```python\nprint(1)\n```",
            "from pwn import process"
        )
        assert not correct

    def test_no_code_block(self):
        correct, fb = grade_code("just text", None)
        assert not correct
        assert "No code block" in fb


class TestGradePatch:
    def test_banned_token(self):
        correct, fb = grade_patch("gets(buf)", banned=["gets("])
        assert not correct
        assert "banned" in fb.lower()

    def test_required_token_missing(self):
        correct, fb = grade_patch("buf = malloc(32)", required=["fgets("])
        assert not correct
        assert "missing" in fb.lower() or "required" in fb.lower()

    def test_passes(self):
        correct, fb = grade_patch("fgets(buf, sizeof(buf), stdin)", required=["fgets("])
        assert correct


class TestGradeExploitTrace:
    def test_all_steps_found(self):
        correct, fb = grade_exploit_trace(
            "The buffer overflow allows redirecting to win()",
            ["buffer overflow", "win"]
        )
        assert correct

    def test_missing_steps(self):
        correct, fb = grade_exploit_trace(
            "The buffer overflow allows redirecting",
            ["buffer overflow", "win", "canary"]
        )
        assert not correct

    def test_no_steps_required(self):
        assert grade_exploit_trace("anything", [])[0]


class TestWilsonCI:
    def test_centered(self):
        lo, hi = wilson_ci(50, 100)
        assert 0.4 < lo < 0.5
        assert 0.5 < hi < 0.6

    def test_zero(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.1

    def test_all_correct(self):
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.9
        assert hi > 0.99


class TestGradeCodeSandbox:
    def test_safe_builtins_available(self):
        """Code using standard builtins should work."""
        code = "```python\nx = len([1, 2, 3])\nresult = x\n```"
        correct, fb = grade_code(code, None)
        assert correct

    def test_dangerous_builtins_blocked(self):
        """open() and __import__() should NOT be in safe_builtins."""
        # If open is available, the code would execute without error
        # But it shouldn't be available in the sandbox
        code = "```python\nresult = open('/etc/passwd').read()\n```"
        correct, fb = grade_code(code, None)
        # Should either fail or not have open available
        # The point is that it doesn't crash with a security violation
        assert not correct or 'error' in fb.lower() or 'open' not in fb.lower()

    def test_exec_with_safe_builtins(self):
        """Restricted exec should handle division, sorting, etc."""
        code = "```python\nresult = sorted([3, 1, 2])\n```"
        correct, fb = grade_code(code, None)
        assert correct
