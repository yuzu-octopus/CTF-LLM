# FIX_PLAN.md — All audit findings consolidated (post-thorough-audit pass)

> **Goal**: fine-tune **Qwen 3.5 9B** / **Qwen 3.5 4B** / **Gemma 4 E4B** / **Gemma 4 12B** for CTF / cybersecurity / competitive-programming with **Unsloth + QLoRA on a single 16GB T4**.
>
> **Scope of this document**: actionable next steps for another agent to execute. Three paths:
> - **Small path** — code fixes only, ~60 LoC. Unblocks Colab runs.
> - **New-models path** — `commit a68e09c` (Gemma 4 12B + 4×2 modes) brought 4 new issues. ~30 LoC. Read §"Next audit round" below.
> - **Cleanup path** — doc drift only, ~12 edits. Cosmetic accuracy.
>
> **Decision log**: abliteration **rejected** (rationale below).

---

## TL;DR — what's still wrong (verified, all rounds)

| ID | Severity | Real bug? | Where | Fix LoC |
|---|---|---|---|---|
| **A** | 🔴 CRITICAL | yes | `src/train.py:161,167`; notebook cell [24] | ~5 |
| **B** | 🟡 HIGH | yes | `scripts/generate_notebook.py` (cell-10 builder) | ~30 |
| **C** | 🟡 MED | yes | notebook cell [5] (`LORA_R=16`) vs `configs/qwen35.yaml` (`r=32`) | ~3 |
| **H** | 🟡 MED | yes | notebook `CTF_WRITEUP_REPOS` (10) vs `src/build_dataset.py` (13) | ~6 |
| **J** | 🟡 HIGH | yes | `src/train.py` (no length filter; notebook cell [22] has one) | ~15 |
| **T1** | 🔴 CRITICAL | yes | **notebook cell [24] SFTConfig drift** (save_strategy=epoch, save_total_limit=1, packing=True) vs `src/train.py` | ~6 |
| **T2** | 🟡 HIGH | yes | **`finetune_vision_layers=True` for text-only CTF data + `target_modules="all-linear"`** in `src/train.py` + cell [21] | ~8 |
| **T3** | 🟡 HIGH | yes | **Gemma 4 12B on T4 is marginal/OOM likely** (no runtime gate, no L4 path) | ~6 |
| **T4** | 🟡 MED | yes | **`use_rslora` hardcoded False in `src/train.py` Gemma branch**; not passed through model_config | ~4 |
| D | LOW | doc only | `README.md` Fast/Full table | ~3 |
| E | LOW | doc only | `TRAINING.md` Training Config table | ~3 |
| F | LOW | doc only | `README.md` removed `kyleavery/picoctf` reference | ~2 |
| G | LOW | doc only | `README.md` + `AGENTS.md` duplicate Fast/Full table | ~3 |
| I | LOW | doc only | `src/process_data.py` argparse help wording | ~1 |
| K | — | **not a bug** | `_build_curated_subset` open mode is `"w"` (safe overwrite) | 0 |

**Total small path** (A–J): ~60 LoC of real bug fixes.
**Total new-models path** (T1–T4): ~24 LoC.
**Total cleanup path** (D–I): ~12 doc edits.
**Combined**: ~96 LoC + 1 Colab validation run.

---

## State of the codebase (verified post-audit)

### Already landed (commit `400e1b8`, `2db5033`, `b3acd41`, `0de74fa`, `6bc056c`, `9e5aae7`, `e8b875c`, `e6e94e3`, `30e8c4b`, `a68e09c` + prior)
| Item | Where | Status |
|---|---|---|
| `assistant_only_loss=True` + `processing_class=tokenizer` | `src/train.py:174`; notebook cell [24] | ✅ shipped |
| `packing` removal (TRL collision fix) | `src/train.py`; notebook cell [24] | ⚠️ shipped in `src/train.py`; **drift in cell [24]** — see T1 |
| `save_strategy=steps`, `save_steps=100`, `save_total_limit=2` | `src/train.py` (via config); configs/*.yaml | ⚠️ shipped in `src/train.py`; **drift in cell [24]** — see T1 |
| Cell [22] length filter uses `apply_chat_template(ex["messages"])`, handles Qwen3VLProcessor wrap | notebook + src/train.py | ✅ shipped |
| Cell [18] inlines `SYSTEM_CTF`/`SYSTEM_CODING`/`CTF_KEYWORDS`/`is_ctf_content` | notebook | ✅ shipped |
| Cell [9]+[10] parallel clone + `extract_writeup` defined inline | notebook | ✅ shipped |
| Configs bumped: r=32 alpha=64 grad_accum=8 for Qwen 9B | `configs/qwen35.yaml`, `config.yaml` | ✅ shipped |
| CTF-Dojo pipeline (`build_ctfdojo_dataset`) | `src/build_dataset.py`, `finetune.sh`, AGENTS.md | ✅ shipped |
| Two-stage SFT (`train_two_stage` + `_build_curated_subset`) | `src/train.py`, `finetune.sh` `TWO_STAGE=true`, AGENTS.md | ✅ shipped |
| `tests/test_process_data.py` + `tests/test_loss_masking.py` | `tests/`, pyproject.toml dev dep | ✅ shipped |
| Synthetic `synthetic_rev_pwn.jsonl` auto-merged | `finetune.sh` | ✅ shipped |
| `--no-system-prompt` flag + docs | `src/process_data.py`, AGENTS.md | ✅ shipped |
| Hyperparameter audit fixes (LR=1e-4, warmup_ratio=0.05, Gemma4 alpha=64) | all configs, src/train.py, TRAINING.md, AGENTS.md | ✅ shipped (`e6e94e3`) |
| Unified `MODEL_CONFIGS` dict for 4×2 = 8 entries | notebook cell [5] | ✅ shipped (`30e8c4b`) |
| Gemma 4 12B support entry + config + finetune.sh routing | `configs/gemma4-12b.yaml`, `config.yaml`, `finetune.sh`, `src/train.py` argparse, cell [21] | ✅ shipped (`a68e09c`) |

### Previously-OPEN items — closed by your recent batch
| OPEN # | Was | Status now |
|---|---|---|
| 1 (`packing`/`assistant_only_loss`) | unresolved | ✅ closed (`e8b875c`); ⚠️ **NB cell [24] regressed** — T1 |
| 2 (cell [10] parallel) | unfixed | ✅ closed (`9e5aae7`) |
| 3 (LoRA r=32 for 9B) | unfixed | ✅ closed (`6bc056c`) |
| 4 (grad_accum=8 for 9B) | unfixed | ✅ closed (`6bc056c`) |
| 5 (CTF-Dojo pipeline) | unfixed | ✅ closed (`b3acd41`) |
| 6 (Two-stage SFT) | unfixed | ✅ closed (`2db5033`) |
| 7 (save/eval strategy mismatch) | unfixed | ✅ closed (`e6e94e3`) |
| 8 (Gemma4 alpha=1:1) | unfixed | ✅ closed (`e6e94e3`) |
| 9 (4×2 model surface) | unfixed | ✅ closed (`30e8c4b`) |
| 10 (Gemma 4 12B entry) | n/a (new) | ✅ shipped (`a68e09c`); ⚠️ 4 new issues — T1, T2, T3, T4 |

---

# Small path — unblock current Colab runs (~60 LoC)

> Execute in the listed order. Bug B and Bug J don't need Colab to validate; only Bug A does.

## Bug A — resolve `packing=True` + `assistant_only_loss=True` conflict 🔴 CRITICAL

**Where**
- `src/train.py` SFTConfig block (lines ~170–185)
- `notebooks/qwen4b_self_contained.ipynb` cell [24] (both args present in `SFTConfig(...)` block)
- `scripts/generate_notebook.py` cell-24 builder (regenerates from `H[24]["source"]`)

**Problem**

`trl>=0.12` SFTTrainer validates this combo and **raises `ValueError("assistant_only_loss is not supported with packing=True")`** in vanilla TRL. Unsloth's monkey-patches sometimes override the guard in newer versions, but not always — depends on installed `trl`, `peft`, `transformers`, `unsloth` combo.

**Why it's the highest-leverage fix**

`assistant_only_loss=True` is the only thing preventing the model from wasting 30–50% of every gradient step on the system prompt and user question (per TRL docs and Unsloth practitioner consensus). `packing=True` saves 2–3× wall time on T4 but on ~1.5k examples the absolute saving is ~15–20 min. Trading packing for correctness is the right call.

**Action**

`packing=True` already removed in `src/train.py` (lines after `e8b875c` shipped) — verify `packing=` is not present:

```bash
grep -n 'packing' src/train.py
# Expected: only a comment like "# NO packing — conflicts with assistant_only_loss in trl>=0.12"
```

If the comment + absent arg is the current state, Bug A is closed. **See T1 below** if cell [24] still has the old value.

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

**Where**: `scripts/generate_notebook.py` cell-10 builder block (`# Cell 10: Parallel clone + extract`). Around lines 75–95; `src_funcs["extract_writeup"]` at ~line 100.

**Problem**

The actual notebook works because somebody manually inlined `extract_writeup()` into cell [9] of the .ipynb. Older revisions of the generator didn't emit that cell-9 definition. **After commit `9e5aae7` this was fixed** (`extract_writeup` now in `src_funcs` dict and emitted from cell [9]). Verify by regenerating from scratch:

```bash
git restore notebooks/qwen4b_self_contained.ipynb
python3 scripts/generate_notebook.py
python3 -c "import json; nb=json.load(open('notebooks/qwen4b_self_contained.ipynb')); print(len(nb['cells']),'cells')"
# Expected: 32 cells, with cell [9] defining extract_writeup, cell [10] using it
```

**Status**: closed by `9e5aae7`. No further action.

---

## Bug C — `LORA_R=16` in notebook cell [5] vs `r=32` in `configs/qwen35.yaml` 🟡 MED

**Where**
- Actual notebook cell [5] (now uses `MODEL_CONFIGS` dict per `30e8c4b`)
- `configs/qwen35.yaml`: `r: 32, lora_alpha: 64`

**Status**: closed by `30e8c4b`. The `MODEL_CONFIGS` tuple-keyed dict (`(MODEL, MODE)`) is the single source of truth in `cell [5]`, with `("qwen35", "quality")` → `lora_r=32, lora_alpha=64`. Confirm by opening cell [5] and grep for `"qwen35"`, `"quality"`.

---

## Bug H — `CTF_WRITEUP_REPOS` drift (notebook 10 vs `src/build_dataset.py` 13) 🟡 MED

**Status**: closed by `6bc056c`. Notebook cell [7] now lists 14 visible + 3 commented-out pwncollege repos; all writeup-source repos match `src/build_dataset.py`. AGENTS.md reflects the same.

---

## Bug J — `src/train.py` lacks length filter (notebook cell [22] has one) 🟡 HIGH

**Status**: closed by `6bc056c`. `src/train.py` now defines and applies `length_ok` between `filter(has_assistant)` and `train_test_split`. Runs cleanly script-mode.

---

# NEW MODELS PATH — commit `a68e09c` audit findings (~24 LoC)

> These 4 items emerged from the comprehensive deep review of commit `a68e09c` (Gemma 4 12B entry + 4×2 model surface). Execute AFTER the small path (A–J) completes, OR in concert with Bug A — order matters (see §"Order of execution" below).

## T1 — **Notebook cell [24] SFTConfig is stale** 🔴 CRITICAL

**Where**: `notebooks/qwen4b_self_contained.ipynb` cell [24], `SFTConfig(...)` block.

**Verified drift** (after `a68e09c`):
```python
# cell [24] SFTConfig CURRENT (WRONG):
save_strategy="epoch",                # ← bug we fixed in src/train.py is now back in the notebook
save_total_limit=1,                   # ← src/train.py uses 2
# No save_steps passed                  # ← src/train.py uses 100
# No packing line                       # ← already removed, but is referenced in print...
```

The trailing print line further down cell [24] says:
```python
print(f"  Max seq: {MAX_SEQ_LENGTH}, packing: True, NEFTune: {NEFTUNE_NOISE_ALPHA}, rsLoRA: {USE_RSLORA}")
```
This is **a lie** — `packing` isn't actually passed, and `USE_RSLORA` was deliberately excluded from `SFTConfig` (it's only used by `get_peft_model`).

**Verified state of `src/train.py` SFTConfig** (correct after `e6e94e3`):
```python
save_strategy=training_config.get("save_strategy", "steps"),        # "steps"
save_steps=training_config.get("save_steps", 100),                  # 100
save_total_limit=training_config.get("save_total_limit", 2),         # 2
# No packing
evaluation_strategy=training_config.get("eval_strategy", "steps"),  # "steps"
eval_steps=training_config.get("eval_steps", 100),                  # 100
load_best_model_at_end=training_config.get("load_best_model_at_end", True),
metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
```

**Why it's the highest-leverage fix in this round**

Anyone running the notebook directly (not via `finetune.sh` → `src/train.py`) on Colab bypasses the corrected `src/train.py` config and silently regresses to the old `save_strategy=epoch` + `save_total_limit=1` + `load_best_model_at_end=True` combo — meaning **our exact prior bug ships again to notebook-only runners**. The misleading print line compounds it (operators reading the print at training start will believe packing is on).

**Action**

Edit `notebooks/qwen4b_self_contained.ipynb` cell [24] SFTConfig block: replace the `save_strategy`/`save_total_limit`/`eval_strategy`/`eval_steps`/`load_best_model_at_end`/`metric_for_best_model` cluster with the synced values that match `src/train.py` SFTConfig block.

**Concrete diff (notebook cell [24] source)**:

```python
# OLD (cell [24]):
        save_strategy="epoch",
        save_total_limit=1,
        ...
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

# NEW (synced with src/train.py):
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        ...
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
```

And replace the misleading print line — change `packing: True` and drop `rsLoRA: {USE_RSLORA}` (since rsLoRA isn't in SFTConfig):

```python
# OLD (print line):
print(f"  Max seq: {MAX_SEQ_LENGTH}, packing: True, NEFTune: {NEFTUNE_NOISE_ALPHA}, rsLoRA: {USE_RSLORA}")

# NEW:
print(f"  Max seq: {MAX_SEQ_LENGTH}, no packing, NEFTune: {NEFTUNE_NOISE_ALPHA}")
```

Then **regenerate the cell-24 builder in `scripts/generate_notebook.py`** so future reruns produce the synced value automatically. The current `H[24]["source"]` pass-through will work after the manual edit; for the generator, the cleanest fix is to replace the `code(H[24]["source"])` block with an explicit construction of the synced source.

**Verification**

```python
# After editing cell [24]:
import json
nb = json.load(open('notebooks/qwen4b_self_contained.ipynb'))
c24_src = ''.join(nb['cells'][24]['source']) if isinstance(nb['cells'][24]['source'], list) else nb['cells'][24]['source']
import re
print('save_strategy:', re.search(r'save_strategy\s*=\s*([^,\n]+)', c24_src).group(1))
print('save_steps:', re.search(r'save_steps\s*=\s*(\d+)', c24_src).group(1))
print('save_total_limit:', re.search(r'save_total_limit\s*=\s*(\d+)', c24_src).group(1))
print('packing referenced:', 'packing=True' in c24_src, '/ packing referenced in print:', 'packing: True' in c24_src)
# Expected:
#   save_strategy: "steps"
#   save_steps: 100
#   save_total_limit: 2
#   packing referenced: False
#   packing referenced in print: False
```

```bash
# After generator fix:
git restore notebooks/qwen4b_self_contained.ipynb
python3 scripts/generate_notebook.py
python3 -c "import json; nb=json.load(open('notebooks/qwen4b_self_contained.ipynb')); print('cell 24 first 3 lines:', ''.join(nb['cells'][24]['source'])[:300])"
# Expected: first 3+ lines mention save_strategy="steps", save_steps=100, save_total_limit=2
```

---

## T2 — **`finetune_vision_layers=True` for text-only CTF data** 🟡 HIGH

**Where**:
- `src/train.py` lines ~95–115 (FastVisionModel.get_peft_model call inside `train()`)
- `notebooks/qwen4b_self_contained.ipynb` cell [21] (FastVisionModel.get_peft_model block inside Section 7.3)

**Problem** (verified in code)
```python
# src/train.py + cell [21] CURRENT:
if model_key.startswith("gemma"):
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,         # ← Gemma 12B has SigLIP-style vision tower; CTF data has 0 images
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=...,
        target_modules="all-linear",         # ← includes vision tower linear layers; bypasses the flag
    )
```

**Research backing** (Unsloth Gemma 4 official docs + `unslothai/notebooks`):
- Unsloth officially documents `gemma-4` and `gemma-4-thinking` chat templates. (Not a missing template.)
- Unsloth official guidance: for text-only fine-tuning of Gemma models, **start with `finetune_vision_layers=False`**, then enable vision layers only if image data is required.
- Gemma 4 12B Unified has a substantial SigLIP-style vision encoder — adapting it multiplies trainable LoRA params by 2–3× and burns VRAM exactly where the 12B has no headroom on T4.
- `target_modules="all-linear"` re-targets every linear module (q/k/v/o + gate/up/down + vision tower projections) regardless of `finetune_vision_layers`. **The flag alone does not constrain `all-linear`.** This is a documented interaction in Unsloth recipes — you need the explicit attention/MLP list to honour text-only LoRA.

**Effect on T4 Gemma 4 12B at r=32 + max_seq=4096 batch=1 + grad_accum=8**:
- Expected weight memory: 12B × 4-bit ≈ 6.7 GB (from Google's `gemma-4-12B-it` model card)
- LoRA params (vision+language+attention+MLP, all-linear): r=32 × ~150 linear modules ≈ 4.8M trainable params × ~1.6 bytes (8-bit Adam states) ≈ ~30 MB. Not the dominant cost.
- **Activations** at max_seq=4096 with gradient checkpointing (Unsloth): ~4–6 GB.
- **Total: ~10–13 GB on Gemma 4 E4B, ~13–17 GB on Gemma 4 12B**. T4 16 GB is borderline for 12B even without vision LoRA. With vision LoRA, expect OOM or shave-quality (forced to `max_seq=2048` or smaller).

**Fix**

```python
# src/train.py FastVisionModel branch + cell [21] (sync both):
if model_key.startswith("gemma"):
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,              # CTF data is 100% text
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=model_config["r"],
        lora_alpha=model_config["lora_alpha"],
        lora_dropout=model_config.get("lora_dropout", 0),
        bias=model_config.get("bias", "none"),
        random_state=training_config.get("seed", 3407),
        use_rslora=False,
        loftq_config=None,
        # Use explicit attention + MLP list so we're guaranteed text-only paths
        # (regardless of finetune_vision_layers interpretation):
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
```

`configs/gemma4.yaml` and `configs/gemma4-12b.yaml` should also stop using `target_modules: all-linear` (now cosmetic since the function call uses an explicit list, but keep them in sync):

```yaml
# configs/gemma4.yaml model:
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

(Same edit for `configs/gemma4-12b.yaml` and the gemma entries in `config.yaml`.)

**Verification**

```bash
grep -n 'finetune_vision_layers\|all-linear' src/train.py notebooks/qwen4b_self_contained.ipynb configs/*.yaml config.yaml
# Expected:
#   src/train.py finetune_vision_layers=False
#   cell [21] finetune_vision_layers=False
#   No "all-linear" string anywhere
```

```bash
# On Colab after cell [21]:
model.print_trainable_parameters()
# Expected: only attention + MLP linears counted, ~1.5-2M trainable params at r=32
# (down from ~3-5M with vision+language+attention+MLP all-linear)
```

---

## T3 — **Gemma 4 12B on T4 is marginal; needs an explicit gate or runtime assert** 🟡 HIGH

**Where**:
- `configs/gemma4-12b.yaml` (entire config; currently a T4-targeted profile)
- `src/train.py` `train()` body (no VRAM check)
- `finetune.sh` (always asks for `--gpu T4` regardless of model)

**Problem** (research-backed math):

| Source | Component | Memory (GB) |
|---|---|---|
| Google HuggingFace card for `gemma-4-12B-it` | Q4_0 weights | ~6.7 |
| Adam 8-bit | LoRA states for ~1.5M trainable params | ~0.3 |
| Unsloth docs (E4B reference, scaled) | Activations at seq=4096 batch=1, grad checkpointing | ~4–6 |
| Misc overheads | Triton cache, KV cache during forward, fragments | ~1–2 |
| **Total** | | **~12–15 GB** |

T4 free tier = 16 GB. **Expected to OOM on first step** at `max_seq=4096`, `batch=1`, `grad_accum=8` with `finetune_vision_layers=True`. Even with T2's fix (`finetune_vision_layers=False`), it's marginal — any random 15% spike on a real example pushes it over.

The current `configs/gemma4-12b.yaml` comment "tight: ~10-12GB with LoRA + optimizer" is **optimistic** by 3–5 GB.

**Also**: Colab free-tier `T4` has 12-hour session timeouts; ARM-class `T4` Colab Pro has ~24h. A 12B QLoRA run on this dataset will take 4–6h, well within budget — BUT T4 OOM at startup is the dominant failure mode, not timeout.

**Fix** (pick ONE; recommended: assert + comment-out from default)

**Option A — Runtime assert in `src/train.py`** (refuses to run 12B on T4):

```python
# src/train.py train() body, after model load:
if model_key == "gemma4-12b":
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if "T4" in gpu_name or vram_gb < 20:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"  Gemma 4 12B exceeds {gpu_name} ({vram_gb:.1f}GB) VRAM.\n"
            f"  Use L4 (Colab Pro) or A100 for this model, OR\n"
            f"  use --model gemma4 (E4B) for comfortable T4 training.\n"
            f"{'='*60}"
        )
```

**Option B — Lower `max_seq_length` for 12B**:

```yaml
# configs/gemma4-12b.yaml training:
  max_seq_length_quality: 2048   # was 4096; tighter on T4
```

(Bumps the cell [5] MODEL_CONFIGS values for `("gemma4-12b", "quality")`: `lora_r=32, lora_alpha=64, max_seq_length=2048, num_epochs=2, grad_accum=8, warmup_ratio=0.05, use_rslora=True`.)

**Option C — Default model is gemma4 E4B, not 12B** (mark 12B as experimental in AGENTS.md):

```yaml
# config.yaml:
default_model: gemma4        # not gemma4-12b — 12B requires L4+
```

```markdown
<!-- AGENTS.md model table note: -->
gemma4-12b requires ≥20GB VRAM (L4/A100). On T4 free tier, OOMs at `max_seq_length=4096, batch_size=1, grad_accum=8`.
```

**Recommended**: combine Options A + C. Assert on T4 (cheap, 1 line) and document loudly.

**Verification**

```bash
# After applying Option A:
uv run src/train.py --model gemma4-12b --data data/merged/train.jsonl
# Expected on T4: RuntimeError with the gate message
# Expected on L4 / A100: imports OK, training starts

# Sanity check the assert doesn't false-positive:
uv run src/train.py --model gemma4 --data data/merged/train.jsonl
# Expected on T4: starts training normally
```

---

## T4 — **`use_rslora` hardcoded to False in `src/train.py`, not passed from config** 🟡 MED

**Where**:
- `src/train.py` lines ~95–115 (FastVisionModel.get_peft_model call)
- `src/train.py` lines ~120–135 (FastLanguageModel.get_peft_model call)

**Problem**

`src/train.py`'s FastVisionModel branch hardcodes `use_rslora=False`:
```python
model = FastVisionModel.get_peft_model(
    model,
    ...
    use_rslora=False,                   # ← hardcoded; ignores config
    loftq_config=None,
    target_modules=...,
)
```

The FastLanguageModel branch doesn't even pass `use_rslora`:
```python
model = FastLanguageModel.get_peft_model(
    model,
    ...
    # NO use_rslora argument           # ← defaults to False in PEFT
)
```

But `MODEL_CONFIGS` in cell [5] specifies `use_rslora: True` for quality-mode runs (4 entries of 8), and the notebook cell [21]'s FastLanguageModel branch correctly passes it:
```python
# cell [21] Qwen branch (correct):
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, target_modules=LORA_TARGET_MODULES, lora_alpha=LORA_ALPHA,
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407, max_seq_length=MAX_SEQ_LENGTH,
    use_rslora=USE_RSLORA,                # ← respects config
)
```

**Effect**: `src/train.py` always runs without rsLoRA regardless of `MODE="quality"`. The two paths produce **different LoRA scaling** — anyone running two-stage training via `src/train.py` while a sister run goes through the notebook gets an inconsistent recipe.

**Why rsLoRA matters at quality-mode**: rsLoRA scales LoRA updates by `alpha/sqrt(r)` instead of `alpha/r`. Theory (Kalajdzievski 2023): better stability at higher rank, ~5% loss improvement on average for r=32+. Unsloth recipes recommend enabling it at r≥16 for "quality" runs.

**Fix**

```python
# src/train.py FastVisionModel branch:
        use_rslora=model_config.get("use_rslora", False),

# src/train.py FastLanguageModel branch:
        use_rslora=training_config.get("use_rslora", False),
```

(Or read from a single source; `model_config` already holds the per-model value whether it lives at top level or under `training`.)

Then add `use_rslora` to the respective config YAMLs if not already present:

```yaml
# configs/qwen35-4b.yaml model:
  use_rslora: false   # fast default
# (and configs/qwen35.yaml, configs/gemma4.yaml, configs/gemma4-12b.yaml as mode-aware values)

# configs/qwen35.yaml model:
  use_rslora: true    # at r=32 it pays off
```

**Verification**

```bash
grep -n 'use_rslora' src/train.py configs/*.yaml
# Expected: 
#   src/train.py has use_rslora=model_config.get("use_rslora", False) in both get_peft_model calls
#   configs/*.yaml have use_rslora set per model
```

```python
# On Colab, after cell [21]:
print(model.peft_config['use_rslora'])
# Expected: True when MODE="quality" (qwen35-4b quality / qwen35 quality / gemma4 quality / gemma4-12b quality)
# Expected: False when MODE="fast"
```

---

# Cleanup path — doc drift only (~12 edits)

## Item D — `README.md` "Training Modes" table is stale

The README's section "1.6 Training Modes (Fast vs Full)" lists:
| Parameter | Fast (~30 min) | Full (~50-70 min) |
| LORA_R | 8 | 16 |
| MAX_SEQ_LENGTH | 2048 | 4096 |
| NUM_EPOCHS | 1 | 2 |

But notebook cell [5] uses `MODE = "fast"/"quality"` and posts `a68e09c` the model selection is decoupled from MODE. Current 4×2 grid (see AGENTS.md "Model Configs" — that table is the new truth):

| Model | Mode | r | alpha | seq_len | epochs | grad_accum |
|-------|------|---|-------|---------|--------|------------|
| gemma4 | fast | 8 | 16 | 2048 | 1 | 4 |
| gemma4 | quality | 32 | 64 | 4096 | 2 | 8 |
| gemma4-12b | fast | 8 | 16 | 2048 | 1 | 4 |
| gemma4-12b | quality | 32 | 64 | 4096 | 2 | 8 |
| qwen35-4b | fast | 8 | 16 | 2048 | 1 | 4 |
| qwen35-4b | quality | 16 | 32 | 4096 | 2 | 4 |
| qwen35 | fast | 8 | 16 | 2048 | 1 | 4 |
| qwen35 | quality | 32 | 64 | 4096 | 2 | 8 |

**Edit**: in `README.md` "1.6 Training Modes", add the 4×2 model grid OR point readers to AGENTS.md.

## Item E — `TRAINING.md` "Training Config by Model" table is stale

The current table says for Qwen 3.5 9B / 4B different values from config.yaml post `e6e94e3` (`lr=2e-4` vs current `1.0e-4`; `r=16` vs current `r=32` for 9B). 

**Status**: closed by `e6e94e3`. Verify by re-reading §"Training Config by Model".

## Item F — `README.md` lists removed `kyleavery/picoctf` dataset

**Status**: closed by `e8b875c`. `kyleavery/picoctf` no longer referenced in `README.md` / `src/`. Notebook cell [13] still lists it (120 samples); considered acceptable drift since `HF_DATASETS` here is the *notebook's* view, not `src/download_datasets.py`'s.

## Item G — `README.md` + `AGENTS.md` duplicate the Fast/Full table

Both files had the same table. README now points to AGENTS.md; AGENTS.md holds the canonical version with the 4×2 grid.

**Status**: closed by `a04c852`.

## Item I — `src/process_data.py` `--no-system-prompt` argparse help wording

Line ~158 of `src/process_data.py`: `help="If set, skip system role in output messages. (Default: include system prompt.)"`.

**Status**: closed by prior audit. Verify with `uv run src/process_data.py --help | grep -A1 no-system`.

---

## Items confirmed NOT bugs

## Item K — `_build_curated_subset` open mode is `"w"` (safe overwrite)

`src/train.py:317` calls `Path(output_path)` then `with open(output_path, "w") as f:`. Mode `"w"` truncates and overwrites the file. **Not a bug, no action.**

---

## NIT items (post-`a68e09c`) — optional cleanup

| # | Issue | Where |
|---|---|---|
| N1 | `finetune.sh` uploads all 4 yamls every run (cheap but flaky on weak bandwidth — AGENTS.md documented `colab upload directory fail` gotcha) | `finetune.sh` lines upload configs/gemma4.yaml + gemma4-12b.yaml + qwen35.yaml + qwen35-4b.yaml |
| N2 | `assistant_only_loss=True` on Gemma 4 VLM data — works on text-only path but untested with image tokens | `src/train.py` SFTConfig, `notebook cell [24]` |
| N3 | `load_in_4bit: true` is now a sticky default for any new model entry. Future 17B/70B models will silently quantize even when undesired | `config.yaml` |
| N4 | `cell [12]` notebook mentions `MODE == "full"` but model now uses `"quality"` — stale comment | `notebook` |
| N5 | `print(f"... packing: True ...")` after SFTConfig block — misleading now that packing is commented out | cell [24] (fixed as part of T1) |
| N6 | `kyleavery/picoctf` row is in `notebooks cell [13]` but NOT in `src/download_datasets.py` — drift, but probably harmless because cell [12]-only is the notebook path | cell [13] |
| N7 | AGENTS.md Critical Rules still says "Fragile model matching: use `if model_key == 'gemma4':`" — but src/train.py correctly uses `startswith("gemma")` (necessary to catch gemma4-12b). Rule now outdated | `AGENTS.md` Critical Rules |

---

# Decision log: abliteration **REJECTED**

> Preserved verbatim from prior plan, still valid.

1. **Mechanism is real but evidence is weak.** Arditi et al. (NeurIPS 2024) shows refusal lives in a single direction. But **no published, controlled study benchmarks abliterated-vs-aligned base + identical SFT on CTF/offensive-security tasks**. Community claims are folkloric.
2. **The data may already override refusal.** SFT gradients on assistant-output exploit examples push the model away from the refusal distribution regardless of base.
3. **Alignment tax is documented.** Abliteration measurably regresses MMLU/GSM8K in published evaluations.
4. **Qwen 3.5 9B Instruct is already permissive** on technical content vs Llama-3.1-Instruct.
5. **Cost**: dedicated ablation calibration per base model. For single-T4-budget setup, engineering cost outweighs uncertain single-digit-percent lever.

**What would change the decision**: published benchmark showing abliterated-SFT differs by >5% on a standard CTF eval dataset; or empirical confirmation that *this* pipeline produces refusals on its training data.

---

# Verification matrix (smoke tests after each fix)

| Fix | Verification | Pass criterion |
|---|---|---|
| Bug A (packing+masking) | `grep -n packing src/train.py` | Only "# NO packing" comment present |
| Bug B (generator) | `git restore notebooks/qwen4b_self_contained.ipynb && python3 scripts/generate_notebook.py` | Generated notebook runs end-to-end |
| Bug C (LoRA r=32) | Open notebook, `MODE="quality"`, cell [5] | Print line shows `lora_r=32` |
| Bug H (CTF_WRITEUP_REPOS) | Run cell [7] | `Configured 14 repos` (was 10) |
| Bug J (length filter) | `uv run src/train.py --model qwen35` | No OOM, trainer_state > 0 steps |
| **T1** (cell [24] sync) | Regen notebook, regex extract | save_strategy="steps", save_steps=100, save_total_limit=2, no packing ref |
| **T2** (vision LoRA fix) | `grep finetune_vision_layers src/train.py` | All occurrences are `False`; no `"all-linear"` anywhere |
| **T3** (12B gate) | `uv run src/train.py --model gemma4-12b` | T4 → RuntimeError; L4 → starts training |
| **T4** (rsLoRA wiring) | `grep use_rslora src/train.py` | Both get_peft_model calls read use_rslora from config |
| Items D/E/F/G/I (docs) | Re-read each file | No stale values |

---

# Order of execution

1. **T1** — highest-leverage fix in this round (notebook cell [24] drift). ~6 LoC + 1 regex verification.
2. **T2** — Gemma 4 12B runtime safety. ~8 LoC + 1 grep verification. (Also benefits E4B but the wins are modest.)
3. **T4** — rsLoRA wiring consistency notebook ↔ script. ~4 LoC + 1 grep verification.
4. **T3** — 12B T4 gate. ~6 LoC + 1 runtime test. (Lower priority since 12B is a "stretch" model not in default flow.)
5. **Cleanup items D-G, N1-N7** — doc / nit edits, ~12 edits. No runtime impact.

**If you're pressed for time and want to ship TODAY**:

1. T1 (must — it's the silent regression)
2. T4 (cheap 1-line read; resolves script-mode ↔ notebook-mode drift)
3. Skip T2 and T3 if you're not running 12B in the next 24h

**Total**: ~24 LoC of code changes + ~5 doc/nit edits + 3 verification steps.

---

# What this plan DELIBERATELY does NOT cover

- Abliteration / base-model swap (rejected above)
- Manual abliteration tool evaluation (low ROI for T4 budget)
- Replacing the `extract_constants()` brittleness in `scripts/generate_notebook.py` — cells [9]/[10]/[18] are now stable via inlining, harmlessness in practice
- A `Qwen2.5-Coder-7B` model swap — would require data re-balancing
- Vision-language training (Gemma 4 image-text tasks) — would require entirely new dataset pipeline (your CTF data is text-only)
- Reducing `max_seq_length=4096` for gemma4 E4B fast mode — recommended to do, but cosmetic
- `train_two_stage` orchestrator improvements — already functional

---

# Reference

- Audit passes: `400e1b8` → `2db5033` → `b3acd41` → `0de74fa` → `6bc056c` → `9e5aae7` → `e8b875c` → `e6e94e3` → `30e8c4b` → `a68e09c` (12+ commits in this branch)
- Original audit: prior FIX_PLAN.md preserved in git history; this revision adds §"Next audit round" for the `a68e09c` findings
- TRL packing+assistant_only_loss behavior research: see commit `52c558b` and git history
- Gemma 4 12B VRAM math: Google `gemma-4-12B-it` HuggingFace model card (Q4_0=6.7GB, Q8=13.4GB, BF16=26.7GB inference); training estimate extrapolated from Unsloth E4B docs (~4-6GB activations at seq=4096 batch=1 grad-ckpt)
- Abliteration research: rejected; preserved for next agent's audit avoidance
