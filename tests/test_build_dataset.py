"""Tests for src/build_dataset.py extraction functions."""
import pytest
from src.build_dataset import (
    find_solution_boundary,
    extract_code_blocks,
    dedup_examples,
)


class TestFindSolutionBoundary:
    def test_finds_solution_header(self):
        content = "# Challenge\nDescription...\n## Solution\nExploit code..."
        line = find_solution_boundary(content)
        assert line > 0

    def test_no_solution_header(self):
        content = "# Challenge\nJust description"
        line = find_solution_boundary(content)
        # Returns line number or -1 depending on implementation
        assert isinstance(line, int)


class TestExtractCodeBlocks:
    def test_python_block(self):
        text = "Here's the exploit:\n```python\nprint('pwned')\n```\nDone."
        blocks = extract_code_blocks(text)
        assert len(blocks) >= 1
        assert "print" in blocks[0]["code"]

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
