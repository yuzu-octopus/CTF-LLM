# FIX_PLAN.md

> Post-implementation plan. **Commit `20bdd36` landed Fixes 1-8 from the prior plan, plus `tests/`.** This document is the lean remainder — bugs that would still crash or silently degrade a Colab run.

## Headline finding

The actual training surface is the **notebook** (`notebooks/qwen4b_self_contained.ipynb`), per `AGENTS.md`: *"Notebook approach for training, not `colab exec` — long-running training hits Colab session timeout."* Commit `20bdd36` fixed `src/train.py` and `src/build_dataset.py`, but **left 3 real bugs in the notebook that will crash on Colab**:

| # | Location | Bug | Severity |
|---|----------|-----|----------|
| A | Notebook cell [22] | `ex["text"]` access — dataset only has `messages` → KeyError on length-filter | 🔴 CRITICAL — crashes training |
| B | Notebook cells [18]/[19] | Generator extracts parens-style constants as empty tuples; cell [19] references undefined names `SYSTEM_CTF`/`SYSTEM_CODING` → NameError | 🔴 CRITICAL — crashes dataset build |
| C | Notebook cell [10] | Sequential clone+extract loop reverts the `ProcessPoolExecutor` from `src/build_dataset.py` | 🟡 MEDIUM — wastes ~5-10 min |

Plus the root-cause: **`scripts/generate_notebook.py` `extract_constants()` only handles triple-quoted strings**, not parens-style. Any time the notebook is regenerated, cells re-loaded are silently broken.

---

## STILL-NEEDED fixes

### Fix A — 🔴 CRITICAL: Notebook cell [22] length filter crashes

**Where**: `notebooks/qwen4b_self_contained.ipynb` cell [22] (regenerated from `scripts/generate_notebook.py` cell-22 block at lines 167-189).

**Actual code in cell [22]**:
```python
def length_ok(ex):
    tokens = actual_tokenizer.encode(ex["text"], add_special_tokens=False)   # ← KeyError
    return len(tokens) <= MAX_SEQ_LENGTH
...
print(f"\nSample:\n{dataset[0]['text'][:300]}...")   # ← KeyError
```

But cell [19] writes only `messages` to `data/merged/train.jsonl`. No `text` field exists. → KeyError on the first row.

**Fix**: Tokenize the messages instead. Replace the `length_ok` body:

```python
def length_ok(ex):
    text = actual_tokenizer.apply_chat_template(ex["messages"], tokenize=False)
    tokens = actual_tokenizer.encode(text, add_special_tokens=False)
    return len(tokens) <= MAX_SEQ_LENGTH
```

And replace the sample-print:

```python
sample_msgs = dataset[0]["messages"]
print(f"\nSample:\n{sample_msgs[-1]['content'][:300]}...")
```

**Source edit**: `scripts/generate_notebook.py` cell-22 block (lines 167-189 of generator). No data shape change needed.

---

### Fix B — 🔴 CRITICAL: Broken system-prompt constants + undefined names

**Where**:
- `scripts/generate_notebook.py:extract_constants()` (lines 51-72) — only handles triple-quoted docstrings
- `src/process_data.py:24-32` — defines `SYSTEM_PROMPT_CTF`/`SYSTEM_PROMPT_CODING` as **parens-style multi-line**, which the extractor truncates to empty tuples

**Actual code in cell [18]** (broken):
```python
SYSTEM_PROMPT_CTF = (
SYSTEM_PROMPT_CODING = (
                        # ← both empty tuples
```

**Actual code in cell [19]** (uses different names → NameError):
```python
sys_prompt = SYSTEM_CTF if ... else SYSTEM_CODING   # ← NameError: name 'SYSTEM_CTF' is not defined
```

Same `SYSTEM_CTF`/`SYSTEM_CODING` are referenced in cells [30]/[31] (inference smoke tests) — those will also fail.

**Fix**: Two options.

**Option B1** (preferred, minimal): Inline the constants in the notebook's cell [18] directly, AND match the names cell [19] uses.

In `scripts/generate_notebook.py`, replace the cell-18 construction (around line 130) with hardcoded content:

```python
code(["# 6.1 System prompts (inlined for self-containment — see src/process_data.py)\n",
      "SYSTEM_CTF = (\n",
      "    \"Expert CTF player. Specialties: pwn, rev, web, crypto, forensics. \"\n",
      "    \"Always reason step-by-step before exploit code.\"\n",
      ")\n",
      "SYSTEM_CODING = (\n",
      "    \"Expert competitive programmer. Optimize first; explain after. \"\n",
      "    \"Security-aware coding.\"\n",
      ")\n",
      ...
```

(Keep `CTF_KEYWORDS`, `is_ctf_content` from src.)

**Option B2**: Fix `extract_constants()` to handle parens-style multi-line. This is fragile (paren counting across tuple/function args, comments, etc.) and not recommended — Option B1 is simpler and removes the abstraction.

---

### Fix C — 🟡 MEDIUM: Sequential clone+extract in notebook cell [10]

**Where**: `notebooks/qwen4b_self_contained.ipynb` cell [10] (regenerated from `H[10]["source"]` in `scripts/generate_notebook.py` line 117).

**Actual code in cell [10]**:
```python
for repo in tqdm(CTF_WRITEUP_REPOS, desc="Cloning repos"):
    if clone_repo(...):
        ...
        for md in tqdm(md_files, ...):
            ex = extract_writeup(...)
```

This is sequential — clones one at a time, extracts one repo at a time. Wastes ~5-10 min.

**Fix**: Replace the cell with a ThreadPoolExecutor (clone, 4 workers) + ProcessPoolExecutor (extract, 4-8 workers) version that mirrors `src/build_dataset.py:build_writeups_dataset`.

Recommended: instead of duplicating logic, **call into src directly** since this notebook already imports `from unsloth` etc.:

```python
# 3.4 Clone all repos in parallel
sys.path.insert(0, "/content")
from src.build_dataset import build_writeups_dataset, CTF_WRITEUP_REPOS as REPOS_DEF
build_writeups_dataset(f"{WORK_DIR}/data/raw/writeups.jsonl", max_per_repo=MAX_PER_REPO)
```

But `src/` must exist on Colab for this — confirm `finetune.sh` `colab upload`s it (it currently only uploads `train.jsonl` and 3 yaml files).

**Source edit**: `scripts/generate_notebook.py` line 117 (`code(H[10]["source"])`) — replace with the `sys.path.insert` + `from src.build_dataset import …` call.

---

## NO LONGER NEEDED (re-verified)

These items were landed in commit `20bdd36` and re-inspected — fully working:

| Former # | Item | Confirmed status |
|----------|------|------------------|
| 1 | Loss masking (`assistant_only_loss=True` + `processing_class=` + messages-native) in `src/train.py` | ✅ Pass. Only needs Fixes A/B to make the *notebook* path match. |
| 2 | Eval split + `load_best_model_at_end=True` in both paths | ✅ Pass. |
| 3 | `save_total_limit: 1` + `eval_strategy: steps` in all 4 YAMLs | ✅ Pass. |
| 4 | System-prompt shortening (620→113 chars; 230→84 chars) in `src/process_data.py` | ✅ Pass *at the source*. Notebook path still broken by Fix B. |
| 5 | `ThreadPoolExecutor` (4) + `ProcessPoolExecutor` (8) in `src/build_dataset.py` | ✅ Pass at source. **Notebook cell [10] still sequential** (Fix C). |
| 6 | `--no-system-prompt` rename + `AGENTS.md` documentation | ✅ Pass. |
| 7 | `tests/test_process_data.py` + `tests/test_loss_masking.py` smoke suite | ✅ Pass (7 tests). |
| 8 | Split HF dataset download from build_dataset.py → `download_datasets.py` | ✅ Pass. |

## Items from the prior plan that have become IRRELEVANT

These were either rendered moot by other fixes, or were deliberate over-design:

- **`--no-system-prompt` flag + docs (Issue 6)**: now fully doc'd in `AGENTS.md` and is the recommended build command there. No further work needed.
- **System-prompt shortening (Issue 4)**: with `assistant_only_loss=True`, system tokens are masked from loss anyway. The remaining QoL win is dataset size only. **Fix B is more impactful** because the notebook path couldn't see the shortened strings at all.
- **Eval-split ablation in FAST mode**: `eval_steps=100` is unconditional; on fast mode (1 epoch, ~200 samples) this means **one** eval pass at step 100. Effectively free. Keep as-is.
- **Tests for build_dataset.py extraction regex**: data quality coverage from eval split already detects bad data. Not worth adding.

---

## What this plan deliberately does NOT address

These were noted from the original 33-issue review but are out-of-scope for "make the Qwen 3.5 9B fine-tune work":

- Consolidating the 7+ near-identical `download_*` functions in `src/download_datasets.py` (cosmetic, ~80 LoC savings).
- Type hints throughout `src/` (no runtime impact).
- Logging module replacing `print()`.
- Removing hard-coded `--max-per-repo 300` in `finetune.sh`.
- Replacing `scripts/generate_notebook.py`'s cell-indexing brittleness (Fix B's Option B1 sidesteps this; the generator as a whole is still fragile but only Fix A/B/C need addressing for this PR).

---

## Order of execution (highest ROI first)

1. **Fix B** — system prompts. Without this, the notebook literally cannot build a dataset.
2. **Fix A** — cell [22] length filter. Crashes on first filter pass.
3. **Fix C** — parallel cell [10]. Performance, not crash. After A+B are landed, ship.

Total LoC: ~30.

## Verification (post-implementation)

| What | How | Expected |
|------|-----|----------|
| Notebook imports cleanly (no NameError) | Open the notebook in Colab, run cells [0]-[19] in order | Cell [19] prints "Generated N ChatML examples" |
| Cell [22] filter does not crash on `ex["text"]` | Run cell [22] | "Dataset (length-filtered): N examples" without KeyError |
| Notebook can train start-to-finish on FAST mode | Run MODE="fast" all 32 cells on a T4 | "TRAINING COMPLETE" + 3 saved formats |

A regression test that regens the notebook and runs cells [18]-[19] as `exec` would catch both A and B at CI time — nice-to-have, doesn't gate this PR.
