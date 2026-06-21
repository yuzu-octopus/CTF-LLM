# Plan 014: Data Pipeline Test Expansion (F4)

**Commit**: `6d7af02`  
**Status**: ✅ DONE (44d12cb)  
**Effort**: M (~3 h)  
**Risk**: MED (tests may need fixtures or mock data)

## Problem

The two data pipeline scripts have minimal or zero test coverage:
- `src/download_datasets.py` (329 LOC) — **0 tests**
- `src/build_dataset.py` (564 LOC) — **51-line test file covering 3 of 15+ functions**

The data pipeline is the most fragile part of the system (hit by silent data loss bugs during extraction, path mismatches, format changes). Without tests, these regressions go undetected until training produces garbage.

## Current State

```bash
$ wc -l tests/test_build_dataset.py
51 tests/test_build_dataset.py
$ ls tests/test_download_datasets.py
ls: tests/test_download_datasets.py: No such file or directory
```

Existing tests cover:
- `find_solution_boundary` — 2 tests
- `extract_code_blocks` — 2 tests
- `dedup_examples` — 2 tests

Not covered:
- `extract_from_writeup`
- `extract_from_docs`
- `extract_from_ctfd`
- `is_ctf_content` (in process_data.py)
- All download/transform functions in download_datasets.py

## Fix

### Step 1: Write tests for build_dataset.py uncovered functions

Create `tests/test_build_dataset_expanded.py`:

```python
"""Expanded tests for src/build_dataset.py."""
import pytest
from src.build_dataset import (
    extract_solution_text,
    extract_code_blocks,
    find_solution_boundary,
    dedup_examples,
    CTF_WRITEUP_REPOS,
)
from src.process_data import is_ctf_content  # defined in process_data.py


class TestIsCtfContent:
    def test_ctf_keywords(self):
        assert is_ctf_content("pwn challenge")
        assert is_ctf_content("rev writeup")
        assert is_ctf_content("CTF exploit")

    def test_non_ctf(self):
        assert not is_ctf_content("hello world")
        assert not is_ctf_content("python tutorial")


class TestExtractSolutionText:
    def test_basic_extraction(self):
        content = "# Challenge\nDesc\n## Solution\nflag{test}\n## Notes\nmore"
        result = extract_solution_text(content)
        assert "flag{test}" in result

    def test_no_solution(self):
        content = "# Just a description"
        result = extract_solution_text(content)
        assert result is None or result == ""


class TestCtfWriteupRepos:
    def test_repos_listed(self):
        assert len(CTF_WRITEUP_REPOS) > 10

    def test_repo_structure(self):
        for repo in CTF_WRITEUP_REPOS:
            assert "url" in repo or "repo" in repo
```

**Note:** These tests assume the functions are importable without triggering GPU deps. If `build_dataset.py` imports modules that require torch, use `pytest.importorskip("torch")` at the top.

### Step 2: Write tests for download_datasets.py

Create `tests/test_download_datasets.py`:

```python
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
        result = extract_qa(sample)
        assert result is not None
        assert len(result) == 3  # instruction, input, output keys

    def test_messages_format(self):
        sample = {
            "messages": [
                {"role": "user", "content": "What is XSS?"},
                {"role": "assistant", "content": "Cross-site scripting is..."},
            ],
        }
        result = extract_qa(sample)
        if result:
            assert "instruction" in result
            assert "output" in result

    def test_empty_input(self):
        assert extract_qa({}) is None


class TestLoadHfWithFallback:
    def test_unknown_dataset(self):
        """Should return empty list for unknown dataset paths, not crash."""
        result = load_hf_with_fallback("nonexistent/dataset")
        assert isinstance(result, list)
```

### Step 3: Refactor import barriers

If `download_datasets.py` has torch/hardware imports at module level, they need to be moved inside functions so tests can import the pure functions without loading GPU deps.

Pattern:
```python
# Bad — fails without GPU:
import torch
from datasets import load_dataset

# Good — deferred import:
def download_hf_dataset(...):
    from datasets import load_dataset
    ...
```

### Step 4: Verify

```bash
# Run all data pipeline tests
uv run python -m pytest tests/test_build_dataset*.py tests/test_download_datasets*.py -v
# Expected: all tests pass

# Run full suite to confirm no regressions
uv run python -m pytest tests/ -v --tb=short
# Expected: all tests pass
```

## Files to Create/Modify

- Create: `tests/test_build_dataset_expanded.py`
- Create: `tests/test_download_datasets.py`
- Modify: `src/download_datasets.py` (if import barriers exist)
