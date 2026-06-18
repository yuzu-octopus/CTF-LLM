# FIX_PLAN.md — All audit findings consolidated (post-thorough-audit pass)

> **Goal**: fine-tune **Qwen 3.5 9B** for CTF / cybersecurity / competitive-programming with **Unsloth + QLoRA on a single 16GB T4**.
>
> **Scope of this document**: actionable next steps for another agent to execute. Two paths:
> - **Small path** — code fixes only, ~60 LoC. Unblocks Colab runs.
> - **Cleanup path** — doc drift only, ~10 edits. Cosmetic accuracy.
>
> **Decision log**: abliteration **rejected** (rationale below).

---

## TL;DR — what's still wrong (verified)

| ID | Severity | Real bug? | Where | Fix LoC |
|---|---|---|---|---|
| **A** | 🔴 CRITICAL | yes | `src/train.py:161,167`; notebook cell [24] | ~5 |
| **B** | 🟡 HIGH | yes | `scripts/generate_notebook.py` (cell-10 builder) | ~30 |
| **C** | 🟡 MED | yes | notebook cell [5] (`LORA_R=16`) vs `configs/qwen35.yaml` (`r=32`) | ~3 |
| **H** | 🟡 MED | yes | notebook `CTF_WRITEUP_REPOS` (10) vs `src/build_dataset.py` (13) | ~6 |
| **J** | 🟡 HIGH | yes | `src/train.py` (no length filter; notebook cell [22] has one) | ~15 |
| D | LOW | doc only | `README.md` Fast/Full table | ~3 |
| E | LOW | doc only | `TRAINING.md` Training Config table | ~3 |
| F | LOW | doc only | `README.md` removed `kyleavery/picoctf` reference | ~2 |
| G | LOW | doc only | `README.md` + `AGENTS.md` duplicate Fast/Full table | ~3 |
| I | LOW | doc only | `src/process_data.py` argparse help wording | ~1 |
| K | — | **not a bug** | `_build_curated_subset` open mode is `"w"` (safe overwrite) | 0 |

**Total small path**: ~60 LoC of real bug fixes.
**Total cleanup path**: ~12 doc edits.
**Combined**: ~75 LoC + 1 Colab validation run.

---

## State of the codebase (verified post-audit)

### Already landed (commit `400e1b8`, `2db5033`, `b3acd41`, `0de74fa` + prior)
| Item | Where | Status |
|---|---|---|
| `assistant_only_loss=True` + `processing_class=tokenizer` | `src/train.py:150,161`; notebook cell [24] | ✅ shipped |
| `packing=True` | `src/train.py:167`; notebook cell [24] | ⚠️ collides with above — see Bug A |
| `eval_strategy="steps"`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `save_total_limit=1` | both surfaces | ✅ shipped |
| Cell [22] length filter uses `apply_chat_template(ex["messages"])`, handles Qwen3VLProcessor wrap | notebook | ✅ shipped (Fix A) |
| Cell [18] inlines `SYSTEM_CTF`/`SYSTEM_CODING`/`CTF_KEYWORDS`/`is_ctf_content` | notebook | ✅ shipped (Fix B) |
| Cell [9]+[10] parallel clone + `extract_writeup` defined inline | notebook | ✅ shipped (Fix C) |
| Configs bumped: r=32 alpha=64 grad_accum=8 for Qwen 9B | `configs/qwen35.yaml`, `config.yaml` | ✅ shipped (OPEN #3, #4 closed) |
| CTF-Dojo pipeline (`build_ctfdojo_dataset`) | `src/build_dataset.py`, `finetune.sh`, AGENTS.md | ✅ shipped (OPEN #5 closed) |
| Two-stage SFT (`train_two_stage` + `_build_curated_subset`) | `src/train.py`, `finetune.sh` `TWO_STAGE=true`, AGENTS.md | ✅ shipped (OPEN #6 closed) |
| `tests/test_process_data.py` + `tests/test_loss_masking.py` | `tests/`, pyproject.toml dev dep | ✅ shipped |
| Synthetic `synthetic_rev_pwn.jsonl` auto-merged | `finetune.sh` | ✅ shipped |
| `--no-system-prompt` flag + docs | `src/process_data.py`, AGENTS.md | ✅ shipped |

### Previously-OPEN items — closed by your recent batch
| OPEN # | Was | Status now |
|---|---|---|
| 1 (`packing`/`assistant_only_loss`) | unresolved | **still active, see Bug A** |
| 2 (cell [10] parallel) | unfixed | ✅ **closed** (cell [9] inlines `extract_writeup`; cell [10] now uses it) |
| 3 (LoRA r=32 for 9B) | unfixed | ✅ closed (`configs/qwen35.yaml` `r=32`) |
| 4 (grad_accum=8 for 9B) | unfixed | ✅ closed (`configs/qwen35.yaml` `grad_accumulation_steps: 8`) |
| 5 (CTF-Dojo pipeline) | unfixed | ✅ closed (`b3acd41`) |
| 6 (Two-stage SFT) | unfixed | ✅ closed (`2db5033`) |

---

# Small path — unblock current Colab runs (~60 LoC)

> Execute in the listed order. Bug B and Bug J don't need Colab to validate; only Bug A does.

## Bug A — resolve `packing=True` + `assistant_only_loss=True` conflict 🔴 CRITICAL

**Where**
- `src/train.py` line 161 (`assistant_only_loss=True`)
- `src/train.py` line 167 (`packing=True`)
- `notebooks/qwen4b_self_contained.ipynb` cell [24] (both args present in `SFTConfig(...)` block)
- `scripts/generate_notebook.py` cell-24 builder (regenerates from `H[24]["source"]`)

**Problem**

`trl>=0.12` SFTTrainer validates this combo and **raises `ValueError("assistant_only_loss is not supported with packing=True")`** in vanilla TRL. Unsloth's monkey-patches sometimes override the guard in newer versions, but not always — depends on installed `trl`, `peft`, `transformers`, `unsloth` combo.

**Why it's the highest-leverage fix**

`assistant_only_loss=True` is the only thing preventing the model from wasting 30–50% of every gradient step on the system prompt and user question (per TRL docs and Unsloth practitioner consensus). `packing=True` saves 2–3× wall time on T4 but on ~1.5k examples the absolute saving is ~15–20 min. Trading packing for correctness is the right call.

**Action**

Two-step on Colab:

**Step 1 — diagnose truth** (single 1-cell Colab notebook test, ~10s):

```python
import trl, peft, transformers, unsloth
print(f"trl={trl.__version__}")
print(f"peft={peft.__version__}")
print(f"transformers={transformers.__version__}")
print(f"unsloth={unsloth.__version__}")
```

**Step 2 — branch on the result**

| Result | Decision | Edit |
|---|---|---|
| Cell [24] runs and begins training | Both work (Unsloth patched). Keep both. | None |
| Cell [24] raises `ValueError` with `assistant_only_loss` in message | TRL guard active. **Drop `packing=True`** in three places. | `src/train.py:167` delete the `packing=True` line; in `scripts/generate_notebook.py` cell-24 block remove `packing=True` (and the `# Speed: pack short samples...` comment); regenerate notebook via `python3 scripts/generate_notebook.py` |

**Why drop `packing`, not `assistant_only_loss`**: loss-masking correctness > 15 min wall-time saving.

**Verification**

```python
# After training completes in cell [25]:
import json
train_log = json.load(open("/content/outputs/trainer_state.json"))
print("Final step:", train_log["global_step"])
print("Loss history length:", len(train_log["log_history"]))
# Expected: global_step > 0, no ValueError during cell [24]
```

---

## Bug B — `scripts/generate_notebook.py` emits cell [10] calling undefined `extract_writeup` 🟡 HIGH

**Where**: `scripts/generate_notebook.py` cell-10 builder block (`# Cell 10: Parallel clone + extract`). Around lines 75–95.

**Problem**

The actual notebook works because somebody manually inlined `extract_writeup()` into cell [9] of the .ipynb. **The generator does not emit that cell-9 definition**. So:

- ✅ Current `notebooks/qwen4b_self_contained.ipynb` — works (cell [9] is there manually)
- ❌ Re-running `python3 scripts/generate_notebook.py` — emits a notebook where cell [10] calls `extract_writeup(...)` → `NameError` at cell 10 execution

This is a regression introduced when the user hand-edited cell [10] for parallelization but didn't teach the generator to emit the matching `extract_writeup` definition.

**Action**

Two options. **Recommended: extract `extract_writeup` into the `src_funcs` dict** so the generator emits both cells [9] and [10] cleanly.

Edit `scripts/generate_notebook.py`:

1. Add `adapt_for_notebook(extract_function("build_dataset", "extract_writeups_from_repo"))` to `src_funcs` dict (rename key to `extract_writeup` to mirror what cell [10] calls):

```python
src_funcs = {
    "clone_repo": ...,
    "find_solution_boundary": ...,
    "extract_code_blocks": ...,
    "extract_writeup": adapt_for_notebook(extract_function("build_dataset", "extract_writeups_from_repo")),
    ...
}
```

2. Replace the current cell-9 + cell-10 constructions:

```python
# Cell 9 was: code(H[9]["source"])  ← BREAKS after regen
# Cell 10 was: code([...calls extract_writeup...])

# NEW Cell 9 (FROM SRC): helper function definitions
code([
    "# 3.3 Extract Q&A from writeups\n",
    src_funcs["extract_writeup"] + "\n",
    "print('Helper function: extract_writeup defined')"
]),

# NEW Cell 10 (FROM SRC): parallel clone + extract
code([r"""# 3.4 Clone all repos in parallel + extract writeups
import concurrent.futures
t0 = time.time()
all_writeups = []
all_clone_results = {}

def _clone_one(repo):
    p = f"{tempfile.gettempdir()}/nb_{repo['name']}"
    return repo, clone_repo(repo["url"], p), p

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(_clone_one, r): r for r in CTF_WRITEUP_REPOS}
    for f in concurrent.futures.as_completed(futures):
        repo, ok, path = f.result()
        all_clone_results[repo['name']] = (ok, path)
        print(f"  {'OK' if ok else 'FAIL'} {repo['name']}")

for repo_name, (ok, repo_path) in all_clone_results.items():
    if ok:
        repo_info = next(r for r in CTF_WRITEUP_REPOS if r['name'] == repo_name)
        count = 0
        md_files = [m for m in list(Path(repo_path).rglob("*.md")) + list(Path(repo_path).rglob("*.MD"))
                   if m.name.lower() != "readme.md"]
        for md in tqdm(md_files, desc=f"  {repo_name}", leave=False):
            ex = extract_writeup(md, repo_info["category"])
            if ex:
                all_writeups.append(ex)
                count += 1
                if count >= repo_info.get("max_per_repo", MAX_PER_REPO):
                    break

elapsed = time.time() - t0
print(f"\nExtracted {len(all_writeups)} writeups in {elapsed:.1f}s")
"""]),
```

3. Done. The mac/linux `build_dataset.extract_writeups_from_repo` function takes `(repo_path, category, repo_name)` but it's a heavier extractor — for the notebook's lightweight extract_writeup, the in-cell-9 definition should mirror the actual notebook cell [9] (1 markdown file → 1 example dict). Suggested body:

```python
def extract_writeup(md_file, category):
    """Convert one markdown file into a Q&A example (lightweight notebook version)."""
    try:
        content = md_file.read_text(errors='ignore')
        if len(content) < 100 or md_file.name.lower() == "readme.md":
            return None

        challenge_name = md_file.stem.replace("-", " ").replace("_", " ").title()
        boundary = find_solution_boundary(content)
        description = re.sub(r'```.*?```', '',
                                '\n'.join(content.split('\n')[:boundary]),
                                flags=re.DOTALL)
        description = re.sub(r'\n{3,}', '\n\n', description).strip()[:MAX_OUTPUT_LEN]
        solution = re.sub(r'\n{3,}', '\n\n',
                            '\n'.join(content.split('\n')[boundary:])).strip()[:MAX_OUTPUT_LEN]

        if not description or len(description) < 50:
            return None

        code_blocks = extract_code_blocks(content)
        for cf in list(md_file.parent.glob("*.py"))[:2] + list(md_file.parent.glob("*.c"))[:2]:
            try:
                code = cf.read_text(errors='ignore')
                if len(code) > 10:
                    code_blocks.append({"lang": cf.suffix[1:] or "text", "code": code})
            except:
                pass

        if code_blocks:
            output = f"## Solution\n\n{solution}\n\n" + "\n\n".join(
                f"```{b['lang']}\n{b['code'][:MAX_OUTPUT_LEN]}\n```" for b in code_blocks[:3])
        else:
            output = solution or f"## Solution\n\n{description}"
        if len(output) < 50:
            return None

        instruction = "Write an exploit/solution for this challenge" if code_blocks else "Explain how to solve this challenge step by step"
        return {
            "instruction": instruction,
            "input": f"Challenge: {challenge_name}\nCategory: {category}\n\n{description}",
            "output": output
        }
    except Exception:
        return None
```

OR simpler: register the function body as a literal cell in the generator rather than extracting it from `src/build_dataset.py` (the notebook version is intentionally simpler than the script version; reusing the script version would over-extract and OOM).

**Verification**

```bash
git restore notebooks/qwen4b_self_contained.ipynb   # discard manual edits
python3 scripts/generate_notebook.py                  # regen from scratch
python3 -c "import json; nb=json.load(open('notebooks/qwen4b_self_contained.ipynb')); print(len(nb['cells']),'cells')"
# Expected: 32 cells, with cell [9] defining extract_writeup, cell [10] using it
```

---

## Bug C — `LORA_R=16` in notebook cell [5] vs `r=32` in `configs/qwen35.yaml` 🟡 MED

**Where**

- Actual notebook cell [5]:
  ```python
  if "4B" in MODEL_NAME:
      LORA_R = 8
      LORA_ALPHA = 16      # alpha/r = 2
  else:  # 9B
      LORA_R = 16
      LORA_ALPHA = 32      # alpha/r = 2
  ```
- `configs/qwen35.yaml`: `r: 32, lora_alpha: 64`

**Problem**

The notebook is the "single source of truth" training surface (per AGENTS.md: "Notebook approach for training, not `colab exec`"). But its hardcoded `LORA_R=16` for 9B contradicts the config file. If a user runs `MODE = "quality"` (the 9B path), they get r=16 even though the config says r=32. ~50% wasted LoRA capacity on the run that needed it most.

**Action**

Edit notebook cell [5] (and the `scripts/generate_notebook.py` cell-5 builder, which currently does `code(H[5]["source"])` — same as before):

```python
# Old:
else:  # 9B
    LORA_R = 16
    LORA_ALPHA = 32      # alpha/r = 2

# New (match configs/qwen35.yaml):
else:  # 9B
    LORA_R = 32
    LORA_ALPHA = 64      # alpha/r = 2
```

**Verification**

Open the notebook, set `MODE = "quality"`, run cell [5], confirm the print line shows `lora_r = 32`.

---

## Bug H — `CTF_WRITEUP_REPOS` drift (notebook 10 vs `src/build_dataset.py` 13) 🟡 MED

**Where**

- `src/build_dataset.py:CTF_WRITEUP_REPOS` lists 13 repos (2 picoCTF + 2 CryptoHack + 3 pwncollege + Adamkadaban + Exploit-Writeups + 0x1earn + reversingBits + RE-MA-Roadmap + Crypto-Cat/CTF + how2exploit)
- Notebook cell [7] lists only 10 (with the 3 pwncollege repos commented out)
- `AGENTS.md` "Writeup repos" table claims 13

**Problem**

The notebook misses 4 valuable repos: 2 RE (reversingBits, RE-MA-Roadmap) and 2 pwn (Crypto-Cat/CTF, how2exploit). These are also documented in AGENTS.md. The notebook-only pipeline misses ~30% of the available writeup signal.

**Action**

Edit notebook cell [7] (and `scripts/generate_notebook.py`'s `code(H[7]["source"])` block, which currently passes it through):

Add the 4 missing repos before `print(f"Configured...")`:

```python
# After the existing Adamkadaban / Exploit-Writeups / 0x1earn entries:
{"url": "https://github.com/mohitmishra786/reversingBits", "name": "reversingBits", "category": "rev", "max_per_repo": 200},
{"url": "https://github.com/x86byte/RE-MA-Roadmap", "name": "re-ma-roadmap", "category": "rev", "max_per_repo": 150},
{"url": "https://github.com/Crypto-Cat/CTF", "name": "cryptocat-ctf", "category": "pwn", "max_per_repo": 200},
{"url": "https://github.com/Bretley/how2exploit_binary", "name": "how2exploit", "category": "pwn", "max_per_repo": 100},
```

Optional: keep pwncollege repos commented out (AGENTS.md confirms they extract 0 examples due to non-standard markdown).

**Verification**

```bash
# In Colab after cell [10] finishes:
print(f"Configured {len(CTF_WRITEUP_REPOS)} repos")
# Expected: 14 (10 + 4 added; pwncollege still commented out)
print(len(all_writeups))
# Expected: ~2500+ examples (vs ~1900 before)
```

---

## Bug J — `src/train.py` lacks length filter (notebook cell [22] has one) 🟡 HIGH

**Where**: `src/train.py` lines 130–138 (the dataset prep block, between `dataset.filter(has_assistant)` and `split = dataset.train_test_split(...)`).

**Problem**

Notebook cell [22] drops samples that exceed `max_seq_length` after tokenization, with:
```python
def length_ok(ex):
    text = actual_tokenizer.apply_chat_template(ex["messages"], tokenize=False)
    tokens = actual_tokenizer.encode(text, add_special_tokens=False)
    return len(tokens) <= MAX_SEQ_LENGTH

before = len(dataset)
dataset = dataset.filter(length_ok, desc="Filter by length")
```

This prevents OOM on long examples when `max_seq_length=4096` is set. **`src/train.py` does NOT do this**. So if a user runs `uv run src/train.py --model qwen35` on a CTF-writeup-heavy dataset, a single 5000-token example will OOM the T4. Notebook cell is safe; script isn't.

**Action**

Port the filter into `src/train.py` between `dataset = dataset.filter(has_assistant)` and the train_test_split call.

```python
# After line ~135 (current code):
dataset = dataset.filter(has_assistant)
print(f"    Dataset size: {len(dataset)} examples")

# NEW: length filter using chat-template length
def length_ok(ex):
    text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens) <= model_config["max_seq_length"]

before_len = len(dataset)
dataset = dataset.filter(length_ok, desc="Filter by length")
print(f"    Length-filtered: {len(dataset)} examples (dropped {before_len - len(dataset)} long samples)")
```

**Verification**

```bash
# After a script-mode run completes:
cat outputs/<model>-ctf/trainer_state.json | python3 -c "import json,sys; s=json.load(sys.stdin); print('global_step:', s['global_step'])"
# Expected: > 0 (no OOM crash mid-run)

# If you don't strip the merged/processed dataset length first, count long examples:
python3 -c "
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('unsloth/Qwen3.5-9B')
ds = load_dataset('json', data_files={'train':'data/merged/train.jsonl'}, split='train')
def too_long(ex):
    t = tok.apply_chat_template(ex['messages'], tokenize=False)
    return len(tok.encode(t, add_special_tokens=False)) > 4096
n_long = sum(too_long(x) for x in ds.select(range(min(100, len(ds)))))
print(f'Long-example rate: {n_long}/100 sampled')
"
```

---

# Cleanup path — doc drift only (~12 edits)

## Item D — `README.md` "Training Modes" table is stale

The README's section "1.6 Training Modes (Fast vs Full)" lists:

| Parameter | Fast (~30 min) | Full (~50-70 min) |
| LORA_R | 8 | 16 |
| MAX_SEQ_LENGTH | 2048 | 4096 |
| NUM_EPOCHS | 1 | 2 |

But the actual notebook cell [5] uses `MODE = "fast"/"quality"`, with current values:
- Fast: r=8, 4B, seq=2048, 1 epoch
- Quality: r=32 (after Bug C fix), 9B, seq=4096, 2 epochs

**Edit**: replace the table and the surrounding `MODE = "full"` → `MODE = "quality"` references.

## Item E — `TRAINING.md` "Training Config by Model" table is stale

The current table says for Qwen 3.5 9B:
- LoRA rank: 16
- LoRA alpha: 16

But `configs/qwen35.yaml` says `r: 32, lora_alpha: 64`. The Training Config table is wrong.

**Edit**: replace the row `Qwen 3.5 9B | 4-bit QLoRA | LoRA rank 16 | LoRA alpha 16 | ...` with `Qwen 3.5 9B | 4-bit QLoRA | LoRA rank 32 | LoRA alpha 64`.

## Item F — `README.md` lists removed `kyleavery/picoctf` dataset

In `README.md` "Also downloads structured datasets from HuggingFace" table:
```
| `kyleavery/picoctf` | 120 samples | picoCTF challenge solutions |
```

But `src/download_datasets.py` no longer references it (was removed). The notebook cell [13] (`HF_DATASETS = [...]`) does include `kyleavery/picoctf`. Acceptable in notebook, **README should remove it OR sync the notebook too**.

**Edit**: remove the row from `README.md` (preferred; notebook already has it).

## Item G — `README.md` + `AGENTS.md` duplicate the Fast/Full table

Both files have essentially the same Fast vs Full table. Pick one as canonical.

**Edit**: keep it in AGENTS.md (where it appeared first and where the build commands live); remove from README.md.

## Item I — `src/process_data.py` `--no-system-prompt` argparse help wording

Line 90:
```python
parser.add_argument("--no-system-prompt", action="store_true", help="Omit system role from output messages. Default: ON (system prompt written).")
```

The "Default: ON" wording reads as if the flag is enabled by default. The CLI truth is: the flag's *argparse default* is `False`, which means by default `--no-system-prompt` is NOT passed and system prompts ARE written.

**Edit**: change help text to:
```python
help="If set, skip system role in output messages. (Default: include system prompt.)"
```

---

# Items confirmed NOT bugs

## Item K — `_build_curated_subset` open mode is `"w"` (safe overwrite)

`src/train.py:317` calls `Path(output_path)` then `with open(output_path, "w") as f:`. Mode `"w"` truncates and overwrites the file. If a previous run wrote stale `curated.jsonl`, it's safely replaced. **Not a bug, no action.**

Audit comment was about: "what if ctfdojo.jsonl is updated on disk but curated jsonl is from a previous run?" Answer: stage 2 will rebuild `curated.jsonl` from current `data/raw/ctfdojo.jsonl` because `_build_curated_subset` re-reads sources each call.

---

# Decision log: abliteration **REJECTED**

> Preserved verbatim from prior plan, still valid.

1. **Mechanism is real but evidence is weak.** Arditi et al. (NeurIPS 2024) shows refusal lives in a single direction. But **no published, controlled study benchmarks abliterated-vs-aligned base + identical SFT on CTF/offensive-security tasks**. Community claims are folkloric; some cited papers could not be independently verified.
2. **The data may already override refusal.** SFT gradients on `{"role":"assistant","content":"<actual exploit>"}` examples push the model away from the refusal distribution regardless of base.
3. **Alignment tax is documented.** Abliteration measurably regresses MMLU/GSM8K in published evaluations.
4. **Qwen 3.5 9B Instruct is already permissive** on technical content vs Llama-3.1-Instruct.
5. **Cost**: dedicated ablation calibration per base model. For a single-user, single-T4-budget setup, the engineering cost outweighs an uncertain single-digit-percent lever.

**What would change the decision**: published benchmark showing abliterated-SFT differs by >5% on a standard CTF eval dataset; or empirical confirmation that *this* pipeline produces refusals on its training data.

---

# Verification matrix (smoke tests after each fix)

| Fix | Verification | Pass criterion |
|---|---|---|
| Bug A (packing+masking) | Add print-statement cell on Colab first: `import trl; print(trl.__version__)`. Branch by output. | Cell [24] begins training without `ValueError` |
| Bug B (generator) | `git restore notebooks/qwen4b_self_contained.ipynb && python3 scripts/generate_notebook.py && python3 -c "import json; print(json.load(open('notebooks/qwen4b_self_contained.ipynb'))['cells'][9]['source'])"` | Output starts with `def extract_writeup(...)` |
| Bug C (LoRA r=32) | Open notebook, set `MODE = "quality"`, run cell [5] | Print line shows `lora_r = 32` |
| Bug H (CTF_WRITEUP_REPOS) | Run cell [7] in Colab | `Configured 14 repos` (was 10) |
| Bug J (length filter) | `uv run src/train.py --model qwen35 --data data/merged/train.jsonl` with patched code | No OOM, `trainer_state.json["global_step"]` > 0 |
| Items D/E/F/G (docs) | Re-read each file | No stale values |
| Item I (argparse help) | `uv run src/process_data.py --help` | Help text says "Default: include system prompt" |

---

# Order of execution

1. **Bug A** — must do first; rest are scoped around it. ~5 LoC + 1 Colab diag cell.
2. **Bug B** — dev-time, no Colab needed. ~30 LoC.
3. **Bug J** — script-mode OOM safety. ~15 LoC + 1 script-mode validation.
4. **Bug C** — 9B quality run correctness. ~3 LoC.
5. **Bug H** — data parity with src/. ~6 LoC + Colab cell [7]+[10] run.
6. **Cleanup items D-G** — doc edits only, no runtime impact. ~10 edits.
7. **Item I** — argparse text. 1 line.

**Total**: ~70 LoC of code changes + ~11 doc edits + 1 Colab diagnosis + 1 Colab training validation run.

---

# What this plan DELIBERATELY does NOT cover

- Abliteration / base-model swap (rejected above)
- Manual abliteration tool evaluation (low ROI for your T4 budget)
- Pseudocode for `train_two_stage`'s full impl of reloading LoRA mid-session — only the `train(...)` orchestrator wrapper exists; full LoRA-merge-and-resume needs separate iteration
- Replacing the `extract_constants()` brittleness in `scripts/generate_notebook.py` — cells [9]/[10]/[18] are now stable via inlining, so the brittleness is harmless in practice
- A `Qwen2.5-Coder-7B` model swap — would require data re-balancing

---

# Reference

- Audit pass: commit `400e1b8` + family (12+ commits in this branch)
- Original audit: prior FIX_PLAN.md preserved in git history
- TRL packing+assistant_only_loss behavior research: see commit `52c558b` and git history
- Abliteration research: rejected; preserved for next agent's audit avoidance
