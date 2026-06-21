"""Tests for src/download_datasets.py."""
import json
import pytest
from pathlib import Path
from src.download_datasets import (
    load_hf_with_fallback,
    extract_qa,
)


class TestExtractQa:
    def test_open_code_reasoning_format(self):
        """Test extraction from OpenCodeReasoning-style format."""
        sample = {
            "question": "Write a function to add two numbers",
            "answer": "def add(a, b): return a + b",
            "messages": [
                {"role": "user", "content": "Add two numbers"},
                {"role": "assistant", "content": "def add(a, b): return a + b"},
            ],
        }
        q, a = extract_qa(sample)
        assert q is not None
        assert a is not None
        assert "Add two numbers" in q or "add two numbers" in q

    def test_messages_format(self):
        sample = {
            "messages": [
                {"role": "user", "content": "What is XSS?"},
                {"role": "assistant", "content": "Cross-site scripting is..."},
            ],
        }
        q, a = extract_qa(sample)
        assert q is not None
        assert a is not None

    def test_empty_input(self):
        q, a = extract_qa({})
        assert q is None
        assert a is None


class TestLoadHfWithFallback:
    def test_unknown_dataset(self):
        """Should raise for unknown dataset paths, not silently misbehave."""
        with pytest.raises(Exception):
            load_hf_with_fallback("nonexistent/dataset")
