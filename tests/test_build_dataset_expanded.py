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
