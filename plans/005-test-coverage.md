# Plan 005: Test Coverage (6, 7)

**Commit**: `5f04de4`  
**Status**: ✅ DONE (9b71fcc)  
**Effort**: M (~2 h)  
**Risk**: LOW (adding tests only)

## Problem

1. **#6**: `eval.py` grading functions have zero test coverage — `grade_flag`, `grade_mcq`, `grade_code`, `grade_patch`, `grade_exploit_trace`, `grade`
2. **#7**: `build_dataset.py` extraction functions have zero test coverage — `find_solution_boundary`, `extract_challenge_description`, `extract_solution_text`, `extract_code_blocks`, `dedup_examples`

## Fix

### Step 1: Create `tests/test_eval.py`

```python
"""Tests for src/eval.py grading functions."""
import pytest
from src.eval import (
    grade_flag, grade_mcq, grade_code, grade_patch,
    grade_exploit_trace, grade, wilson_ci,
)


class TestGradeFlag:
    def test_exact_match(self):
        assert grade_flag("flag{abc123}", "flag{abc123}") == (True, "Matched: flag{abc123}")

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
        assert hi == 1.0
```

### Step 2: Create `tests/test_build_dataset.py`

```python
"""Tests for src/build_dataset.py extraction functions."""
import pytest
from src.build_dataset import (
    find_solution_boundary,
    extract_challenge_description,
    extract_solution_text,
    extract_code_blocks,
    dedup_examples,
)


class TestFindSolutionBoundary:
    def test_finds_solution_header(self):
        content = "# Challenge\nDescription...\n## Solution\nExploit code..."
        desc, sol = find_solution_boundary(content)
        assert "Description" in desc
        assert "Exploit" in sol

    def test_no_solution_header(self):
        content = "# Challenge\nJust description"
        desc, sol = find_solution_boundary(content)
        assert "Challenge" in desc
        assert sol == ""


class TestExtractCodeBlocks:
    def test_python_block(self):
        text = "Here's the exploit:\n```python\nprint('pwned')\n```\nDone."
        blocks = extract_code_blocks(text)
        assert len(blocks) >= 1
        assert "print" in blocks[0]

    def test_no_code(self):
        blocks = extract_code_blocks("just text")
        assert len(blocks) == 0


class TestDedupExamples:
    def test_removes_duplicates(self):
        examples = [
            {"input": "q1", "output": "a1"},
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ]
        result = dedup_examples(examples)
        assert len(result) == 2

    def test_keeps_unique(self):
        examples = [
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ]
        result = dedup_examples(examples)
        assert len(result) == 2
```

## Verification

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Check coverage (if pytest-cov is installed)
uv run python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Files to Create

- `tests/test_eval.py` (NEW)
- `tests/test_build_dataset.py` (NEW)
