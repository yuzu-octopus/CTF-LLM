"""Inspect a real data sample to see what the trainer will receive."""
import json
from pathlib import Path

merged = Path("data/merged/train.jsonl")
if merged.exists():
    with open(merged) as f:
        lines = f.readlines()
    print(f"Merged: {len(lines)} lines")
    print(f"\nFirst example (raw JSON, truncated to 1500 chars):")
    line = lines[0]
    print(line[:1500])
    print(f"\nKeys: {list(json.loads(line).keys())}")
    ex = json.loads(line)
    if "messages" in ex:
        print(f"\nMessage structure:")
        for m in ex["messages"]:
            print(f"  [{m['role']}] len={len(m['content'])} sample={m['content'][:100]!r}")
else:
    print("no merged data")
    proc = Path("data/processed")
    if proc.exists():
        for f in sorted(proc.glob("*.jsonl")):
            with open(f) as fp:
                lines = fp.readlines()
            print(f"\n=== {f.name}: {len(lines)} lines ===")
            print(f"First ex: {lines[0][:600]}")
            break
