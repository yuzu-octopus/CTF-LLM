"""Dump notebook contents to a project-local file (auditor)."""
import json
from pathlib import Path
NB = Path("notebooks/qwen4b_self_contained.ipynb")
nb = json.load(open(NB))
cells = nb["cells"]
out = Path("scripts/_nb_dump.md")
with open(out, "w") as f:
    f.write(f"# Notebook dump: {NB.name} ({len(cells)} cells)\n\n")
    for i, c in enumerate(cells):
        src = c["source"]
        text = "".join(src) if isinstance(src, list) else src
        f.write(f"\n## [{i}] {c['cell_type']} ({len(text)} chars)\n\n```\n{text}\n```\n\n")
print(f"wrote {out}")
