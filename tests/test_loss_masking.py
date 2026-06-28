from datasets import Dataset
from trl import SFTConfig


def test_build_curated_subset_creates_file():
    """Verify _build_curated_subset creates a valid JSONL file."""
    from pathlib import Path
    import tempfile
    import json as _json
    from src.train import _build_curated_subset
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
        tmp_path = f.name
    _build_curated_subset(tmp_path)
    assert Path(tmp_path).exists()
    with open(tmp_path) as f:
        lines = [l for l in f if l.strip()]
        assert len(lines) > 0, "Curated subset should have >0 examples"
        first = _json.loads(lines[0])
        assert "instruction" in first or "task" in first
        assert "output" in first or "trajectory" in first
    Path(tmp_path).unlink()
