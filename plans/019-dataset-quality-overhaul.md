# Plan 019: Dataset Quality Overhaul

**Commit**: `0831d17`
**Status**: TODO
**Effort**: M (~3 h)
**Risk**: MEDIUM (data regen changes training data composition)

## Problems

15 issues in data pipeline. Critical ones:

**D1 — 76% of writeup training data has wrong system prompt**: 2120/2783 writeup examples from a previous extraction run lack `category` and `source` fields. When processed, `category=""` means `is_ctf_content("")` returns False → they get `SYSTEM_PROMPT_CODING` instead of `SYSTEM_PROMPT_CTF`.

**D2 — Per-repo dedup misses cross-repo duplicates**: `dedup_examples()` is called per-repo, not globally. The same challenge writeup across `Adamkadaban/CTFs` AND `Crypto-Cat/CTF` is kept twice. From 2783 lines, only 664 are unique by `input` hash.

**D3 — Silent crash in error handler**: `process_data.py:111` references undefined variable `i`, so ALL parse error warnings are silently swallowed.

**D4 — Docs data lacks category field**: 13 docs entries from documentation scraper have no `category`, getting wrong system prompt.

**D5 — CTF keyword list too narrow**: `"picoctf"` doesn't match any keyword, so picoCTF writeups get CODING prompt.

**D6 — Gen_eval_bench docstring says N=200, actual is N=210**: Outdated.

## Fix

### Step 1: Fix the crash in process_data.py error handler

`src/process_data.py:111`:
```diff
- print(f"  Warning: skipped line {i+1}: {str(e)[:100]}")
+ print(f"  Warning: skipped line {count + skipped + 1}: {str(e)[:100]}")
```

Also fix the error handler to show useful info even if the format changes:
```python
except Exception as e:
    skipped += 1
    if skipped <= 5:
        print(f"  Warning [line {count + skipped}]: {str(e)[:100]}")
    continue
```

### Step 2: Regenerate writeups.jsonl with correct format

The stale 2120 entries need to be replaced. Either:
(a) Delete `data/raw/writeups.jsonl` and re-run `build_dataset.py --source writeups`
(b) Or write a migration script that re-processes old entries

Option (a) is simpler:
```bash
rm -f data/raw/writeups.jsonl
uv run src/build_dataset.py --source writeups --output-dir data/raw
```

Then re-process and merge:
```bash
uv run src/process_data.py --input data/raw --output data/processed --no-system-prompt
uv run src/process_data.py --merge --input data/processed --output data/merged
```

### Step 3: Add global dedup after per-repo dedup

In `src/build_dataset.py`, after the per-repo loop in `main()`, add a global dedup pass:

```python
# After collecting all writeup examples, do global dedup
all_writeups = []
for repo_result in repo_results:
    if repo_result is not None:
        all_writeups.extend(repo_result)
all_writeups = dedup_examples(all_writeups)  # global dedup
```

This requires changing `main()` to collect results from all repos first, then dedup once globally.

### Step 4: Fix docs data category

In `src/build_dataset.py:scrape_documentation()`, add category classification based on the doc URL:

```python
# Classify documentation by URL
doc_url = item.get("url", "")
if any(kw in doc_url for kw in ["pwntools", "pwn", "rop", "gadget", "angr", "exploit"]):
    item["category"] = "pwn"
elif any(kw in doc_url for kw in ["z3", "sympy", "crypto"]):
    item["category"] = "crypto"
else:
    item["category"] = "coding"
```

This ensures pwntools/angr/ROPgadget docs get `SYSTEM_PROMPT_CTF` instead of CODING.

### Step 5: Expand CTF keyword list

In `src/process_data.py:31`:
```python
CTF_KEYWORDS = [
    "pwn", "rev", "web", "crypto", "ctf", "exploit", "vuln", "shellcode",
    "forensics", "osint", "misc", "picoctf", "reverse engineering",
    "binary exploitation", "xss", "sqli", "injection", "buffer overflow",
    "format string", "rop chain", "heap", "use-after-free",
]
```

### Step 6: Fix gen_eval_bench.py docstring

`src/gen_eval_bench.py:2`:
```diff
- """Generate CTF evaluation benchmark dataset — N=200 (50 per category)."""
+ """Generate CTF evaluation benchmark dataset — N=210 (4 categories × 50 + 10 special)."""
```

### Step 7: Verify

```bash
# Run the data pipeline
rm -f data/raw/writeups.jsonl
uv run src/build_dataset.py --source writeups --output-dir data/raw --max-per-repo 10
# Expected: writeups.jsonl created with category+source fields

# Check all entries have category
python3 -c "
import json
with open('data/raw/writeups.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
no_cat = [l for l in lines if not l.get('category')]
print(f'Total: {len(lines)}, No category: {len(no_cat)}')
assert len(no_cat) == 0, f'{len(no_cat)} entries missing category'
"

# Process and check system prompt assignment
uv run src/process_data.py --input data/raw --output data/processed --no-system-prompt
uv run src/process_data.py --merge --input data/processed --output data/merged

# Verify the error handler fix
python3 -c "
import ast, astor
with open('src/process_data.py') as f:
    tree = ast.parse(f.read())
print('process_data.py compiles OK')
"

# Run tests
uv run python -m pytest tests/ -v --tb=short
# Expected: all pass
```

## Files to Modify

- `src/process_data.py` (fix error handler, expand keywords)
- `src/build_dataset.py` (add global dedup, fix docs category)
- `src/gen_eval_bench.py` (fix docstring)
