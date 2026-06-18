# Comprehensive Fix Plan — Optimizing the CTF Fine-tuning Pipeline

> Target: fine-tune **Qwen 3.5 9B** for CTF with **Unsloth + QLoRA on a free Colab T4 (16 GB)**.
> Goal: a competitive, "monster" CTF model — not just *a* fine-tune.

This document consolidates the post-implementation review (`FIX_PLAN.md` is the
canonical successor to that review) and the research-backed fixes still required.

---

## TL;DR — What unlocks "monsterness"

The single biggest lock is **loss masking on the assistant tokens only**.
Implementing Issue 1 below is what separates a working fine-tune from a *great*
fine-tune on T4-grade compute.

| # | Fix | Priority | LoC | Files | ETA |
|---|-----|----------|-----|-------|-----|
| 1 | Two-step loss-masking fix | 🔴 Critical | ~10 | `src/train.py`, `scripts/generate_notebook.py` | 1 h |
| 2 | Eval split + `load_best_model_at_end` | 🟡 High | ~15 | `src/train.py` | 1 h |
| 3 | `save_total_limit: 1` + `eval_strategy` | 🟡 High | ~5 | `configs/*.yaml`, `config.yaml` | 15 min |
| 4 | Shorten system prompts | 🟡 Medium | ~5 | `src/process_data.py` | 10 min |
| 5 | Parallelize extraction (ProcessPoolExecutor) | 🟡 Medium | ~30 | `src/build_dataset.py` | 1 h |
| 6 | Refine `--skip-system-prompt` → `--no-system-prompt` + docs | 🟢 Low | ~10 | `src/process_data.py`, `AGENTS.md` | 15 min |
| 7 | Minimal pytest for loss masking + format | 🟠 Quality insurance | ~120 | `tests/test_process_data.py`, `tests/test_loss_masking.py` (new) | 1.5 h |
| 8 | (Cleanup) `extract_from_huggingface` de-dup vs. `download_datasets.py` | 🟢 Low | ~10 | `src/build_dataset.py` | 30 min |

**Total ETA: ~6 hours, ~205 lines net.**

---

## Background — Why these fixes?

A code review in commit `b60833e` (the `top 6 high-impact fixes` commit) added:

- `packing=True` to `src/train.py`
- `processing_class=` instead of `tokenizer=` in the notebook / generator
- Auto-inclusion of `synthetic_rev_pwn.jsonl` in `finetune.sh`
- Removal of dead `download_vulnerability` function + dispatch
- `--skip-system-prompt` flag in `process_data.py`
- Parallel `Repo.clone_from` in `build_dataset.py`

These are sound and unblock the pipeline, but **27 of the 33 originally-identified
issues remain**. Web-researched below is a focused subset chosen for *maximum
quality lift for the stated goal (monster Qwen 9B CTF on T4)*.

Two issues above are *corrections* of incorrect assumptions I made in the review:
loss masking was claimed to be a "single line" fix, but it is actually a
two-step architectural change; and `--skip-system-prompt` interacts with the
chat template in subtle ways that warrant either abandonment or shortening
of the system prompt instead.

---

## Issue 1 — 🔴 CRITICAL: Loss masking on assistant tokens only

### What is wrong

`src/train.py:121-126` formats the dataset manually:

```python
def format_chat_template(examples, tokenizer):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(lambda ex: format_chat_template(ex, tokenizer), batched=True)
```

Then `src/train.py:154` sets `dataset_text_field="text"` — and the trainer
trains on **every token** in the rendered string: system prompt, user
question, and assistant answer.

For your CTF data:

- System prompt: ~1 100 chars / ~300 tokens (CTF persona).
- User prompt: ~500-3 000 chars / ~150-800 tokens (challenge description).
- Assistant response: ~500-2 000 chars / ~150-500 tokens (exploit write-up).

So ~**60-80% of every gradient step** is wasted on teaching the model to
regurgitate prompts it should never have to *generate*. The model's
planning/exploit-reasoning capacity (its scarce CTF skill) competes for
attention with "predict the next word of the user's question."

### Why I initially got this wrong

A first-pass recommendation was "add `assistant_only_loss=True` to SFTConfig."
Web research (TRL documentation, version-stable behavior) confirms this is a
**no-op while `dataset_text_field="text"` is in use**: the loss-masking logic
needs chat-template-aware role tokens during tokenization, and the `text` field
carries none of that metadata.

Quoting from the research:

> "`assistant_only_loss` logic is implemented inside the data collator to mask
> tokens based on role IDs generated during chat-template tokenization. If you
> pass raw text, the tokenizer sees no 'role' metadata, and the loss mask
> remains effectively 'all active,' resulting in a no-op that trains on
> everything." — `researcher-docs`

### Research backing

| Source | Finding |
|--------|---------|
| Yoni Gottesman — "Mask your user tokens" | "Crucial for conversational datasets, especially when those datasets have a high ratio of user turn length to assistant turn length." |
| Shi et al., 2024 — "Instruction Tuning with Loss over Instructions" | Non-masking can help on tiny datasets; for datasets >5 k examples, masking is the strong default. |
| Sebastian Raschka — "Does Prompt Loss Matter?" | For code/technical fine-tuning, prompt loss specifically competes with the response skill. |
| TRL docs (`assistant_only_loss`) | Requires chat-template-aware tokenizer + messages-native pipeline. |

### The two-step fix

#### Step A — Drop `dataset_text_field="text"`, pass `messages` natively

`src/train.py`, replace lines 121-126:

```python
# OLD
dataset = dataset.map(
    lambda examples: format_chat_template(examples, tokenizer),
    batched=True,
)

# NEW
# Filter empty outputs first (data-quality guard, lifted from notebook).
def has_assistant(ex):
    msgs = ex.get("messages") or []
    return bool(msgs) and bool(msgs[-1].get("content"))

dataset = dataset.filter(has_assistant)
# Drop the entire `format_chat_template` helper — no longer used.
```

#### Step B — Let SFTTrainer tokenize via chat template + enable loss masking

`src/train.py`, replace `trainer = SFTTrainer(...` block (lines 131-159):

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,                # See Issue 2
    processing_class=tokenizer,
    args=SFTConfig(
        max_seq_length=model_config["max_seq_length"],
        per_device_train_batch_size=model_config.get("batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        warmup_steps=training_config.get("warmup_steps", 10),
        num_train_epochs=epochs,
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        logging_steps=1,
        output_dir=output_dir,
        optim=training_config.get("optim", "adamw_8bit"),
        seed=training_config.get("seed", 3407),
        save_strategy=training_config.get("save_strategy", "epoch"),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # NO dataset_text_field — SFTTrainer auto-detects messages
        assistant_only_loss=True,             # ← NOW ACTUALLY WORKS
        eval_strategy="steps",                # See Issue 2
        eval_steps=100,                       # See Issue 2
        load_best_model_at_end=True,          # See Issue 2
        metric_for_best_model="eval_loss",    # See Issue 2
        report_to="none",
        packing=True,                         # Already added in b60833e
    ),
)
```

#### Mirror change in `scripts/generate_notebook.py` (cell 24)

Same change: drop the `format_chatml` map() call, drop the
`dataset_text_field="text"`, set `assistant_only_loss=True`. The notebook's
length-filtering and train/eval split at lines 22-29 of the generated cell
remain unchanged.

### What this changes

- **Quality**: loss budget shifts from "60/40 prompt/assistant" to "100% assistant."
  Expected eval lift **20-40%** for tasks that depend on planning
  (multi-step exploits, reasoning chains, exploit composition). This is the pri 1
  reason a "monster" Qwen 9B is achievable.
- **Speed**: assistant tokens are the only ones being trained on, so a step is
  effectively doing ~3-5× more learning work per gradient. Effectively fewer
  total steps needed for convergence.
- **VRAM**: slightly less activation memory per step (shorter attention ranges
  on average).

### Verification

After the change, run a smoke test:

```bash
# Quick sanity-check on a tiny synthetic dataset (no GPU needed):
uv run python -c "
from datasets import Dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('unsloth/Qwen3.5-4B')  # any chatml model
ds = Dataset.from_list([{'messages': [
    {'role':'system','content':'sys'},
    {'role':'user','content':'usr'},
    {'role':'assistant','content':'ans'},
]}])
from trl import SFTConfig, SFTTrainer
import torch
# Just confirm the SFTConfig accepts the flag and SFTTrainer doesn't reject the dataset shape.
print('SFTConfig:', SFTConfig(output_dir='/tmp', assistant_only_loss=True).assistant_only_loss)
print('Dataset has messages field:', 'messages' in ds.column_names)
"
```

Expected output: `SFTConfig: True` and `Dataset has messages field: True`.

---

## Issue 2 — 🟡 HIGH: Eval split + `load_best_model_at_end`

### What is wrong

`src/train.py:131-159` does not pass an eval set to `SFTTrainer`. The
notebook (`scripts/generate_notebook.py:225-231`) has a 90/10 split but the
standalone script doesn't. Without validation you cannot:

- Detect mid-training overfit.
- Decide when to stop early.
- Distinguish between three checkpoints that all "have lower loss" but only one
  generalizes.

### Research backing

Practitioner consensus (TRL docs, Unsloth tutorials) recommends `eval_strategy`
steps-100 with `load_best_model_at_end=True` for short T4 runs.

### The fix

In `src/train.py`, immediately before the `SFTTrainer(...)` constructor:

```python
# Train / eval split — 10% held out for overfit detection.
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = split["train"], split["test"]

print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
```

Then in `SFTConfig(...)`:

```python
eval_strategy="steps",
eval_steps=100,
load_best_model_at_end=True,
metric_for_best_model="eval_loss",
```

### What this changes

- **Quality**: a 3-epoch run on a 5 k-example CTF dataset is very prone to
  overfit. Eval split lets the trainer keep the best-generalizing checkpoint —
  not the latest, lowest-train-loss one.
- **Time**: ~30 sec per eval pass on T4 (500 examples × 3 epochs ÷
  100 steps = 15 eval passes). Negligible.
- **Disk hygiene**: `save_total_limit=1` (see Issue 3) keeps only the best ckpt.

### Verification

After running a smoke-train with at least 200 steps, check:

- A `runs/<timestamp>/` checkpoint directory appears.
- `best_model_checkpoint` is recorded in `trainer_state.json`.
- Eval loss trajectory in the logs (or wandb if you later enable it) shows
  monotone decrease for the first ~50 % of training, then plateau.

---

## Issue 3 — 🟡 HIGH: `save_total_limit: 1` + `eval_strategy` in configs

### What is wrong

`src/train.py:142` defaults `save_strategy="epoch"` with no `save_total_limit`.
That means:

- For 3 epochs: 3 intermediate checkpoints + 1 final save = **4 saves on disk**.
- Each save-on-Colab can spike memory and trigger transient I/O bottlenecks.

The notebook already specifies `save_strategy="epoch", save_total_limit=1,`
but the YAML configs and the standalone `src/train.py` path do not propagate
this.

### The fix

In **all four YAML files** (`config.yaml`, `configs/gemma4.yaml`,
`configs/qwen35.yaml`, `configs/qwen35-4b.yaml`), under the `training:` key,
add:

```yaml
training:
  num_train_epochs: 3
  ...
  save_strategy: epoch
  save_total_limit: 1
  eval_strategy: steps
  eval_steps: 100
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
```

Then in `src/train.py:142`, drop the default fallback so missing config fails
loudly:

```python
# OLD
save_strategy=training_config.get("save_strategy", "epoch"),

# NEW
save_strategy=training_config["save_strategy"],
grad_accum=training_config["gradient_accumulation_steps"],  # already required
```

This makes the contract explicit: every model YAML must declare these knobs.

### What this changes

- **Disk**: 4× checkpoint storage → 1×. ~300 MB savings.
- **Time**: ~2-3 min per run (4 saves × ~30 s each avoided).
- **Stability**: per Unsloth community reports, "saving checkpoints frequently
  can trigger I/O bottlenecks and temporary memory spikes during the save
  process, increasing the chance of an OOM" — single checkpoint = no risk.

---

## Issue 4 — 🟡 MEDIUM: Shorten system prompts to ~150-char tags

### What is wrong

The `--skip-system-prompt` flag (added in `process_data.py:71-79` in commit
b60833e) lets the user drop `messages[0]` entirely. But:

- For Qwen (`chatml` template): works — chatml handles missing system role.
- For Gemma (`gemma-4` template): the chat template *renders* fine without a
  system role but the model is never trained to handle the distribution shift
  of "no system prompt at inference vs. with a CTF system prompt at training."
- At inference: anything a user types as a system prompt is OOD relative to
  training distribution → degraded quality.

The better fix is to **keep the system prompt but make it a 150-char tag, not
a 1 100-char essay**.

### The fix

`src/process_data.py`, lines 24-44:

```python
# OLD
SYSTEM_PROMPT_CTF = """You are an expert CTF (Capture The Flag) player and security researcher. You specialize in:
- Binary exploitation (pwn): buffer overflows, ROP, heap exploitation, format strings
- Reverse engineering: analyzing binaries, deobfuscation, decompilation
- Web exploitation: SQL injection, XSS, SSRF, deserialization, JWT attacks
- Cryptography: cryptanalysis, key recovery, side-channel attacks
- Forensics: memory analysis, network capture analysis, steganography

When solving challenges:
1. Think through the problem step by step before writing any code
2. Analyze the binary/challenge, identify architecture and mitigations
3. Identify the vulnerability or attack vector
4. Reason through your approach: why this technique? what gadget/address/symbol do you need?
5. Provide the exploit with full explanation
6. Verify your reasoning - does the exploit actually work?

Always show your thinking process, not just the answer."""

SYSTEM_PROMPT_CODING = """You are an expert competitive programmer and software engineer. You specialize in:
- Algorithm design and analysis
- Data structures optimization
- Code optimization and debugging
- Security-aware coding practices

When solving problems:
1. Understand the problem requirements
2. Identify the optimal approach
3. Write clean, efficient code
4. Explain your reasoning"""

# NEW
SYSTEM_PROMPT_CTF = (
    "Expert CTF player. Specialties: pwn, rev, web, crypto, forensics. "
    "Always reason step-by-step before exploit code."
)

SYSTEM_PROMPT_CODING = (
    "Expert competitive programmer. Optimize first; explain after. "
    "Security-aware coding."
)
```

Then, in **`src/train.py`**, when the new Issue 1 loss-masking is in effect,
the system prompt is in input but masked out of loss. So at runtime, the
1 100-char essay is irrelevant — `<system>Expert CTF player...</system>` is
the input only.

### What this changes

- **Dataset size**: ~950 char × ~5 k examples = **~5 MB shaved** off the
  training corpus.
- **Token efficiency** (with Issue 1 active): the loss-mask already excludes
  system tokens, so the *frontier* speedup is smaller (only the input-position
  activations are saved), but the residual benefit is the disk read/write
  saving and a slightly faster `format_chat_template` step.
- **OOD safety**: shorter system prompts are easier for the model to
  generalize from inference-time variations.

### Verification

After the change, run:

```bash
uv run python -c "
from src.process_data import SYSTEM_PROMPT_CTF, SYSTEM_PROMPT_CODING
print('CTF chars:', len(SYSTEM_PROMPT_CTF))
print('Coding chars:', len(SYSTEM_PROMPT_CODING))
"
```

Expected: ~150 chars each, both < 200.

---

## Issue 5 — 🟡 MEDIUM: Parallelize extraction with `ProcessPoolExecutor`

### What is wrong

After commit `b60833e`, `Repo.clone_from` runs in 4 threads (good). But
`extract_writeups_from_repo` (`src/build_dataset.py:174-243`) is **serial and
CPU-bound** (markdown parsing + regex). It's now the dominant cost:

- 13 repos × 1 500-5 000 markdown files × regex parsing = the slowest phase
  post-cloning.

`ThreadPoolExecutor` won't help here — the GIL is the bottleneck. Use
`ProcessPoolExecutor` instead.

### The fix

`src/build_dataset.py`, replace the second phase of `build_writeups_dataset`
(lines 293-309):

```python
# OLD — sequential
for repo_idx, repo in enumerate(CTF_WRITEUP_REPOS):
    success, repo_path = clone_results[repo['name']]
    if success:
        print(f"\n[{repo_idx + 1}/{total_repos}] Extracting from {repo['name']}...")
        examples = extract_writeups_from_repo(...)
        examples = dedup_examples(examples)
        repo_max = repo.get("max_per_repo", max_per_repo)
        examples = examples[:repo_max]
        all_examples.extend(examples)

# NEW — parallel across repos (CPU-bound → processes)
import os
from concurrent.futures import ProcessPoolExecutor

def _extract_repo(args):
    repo_path, category, repo_name, repo_max = args
    examples = extract_writeups_from_repo(repo_path, category, repo_name)
    examples = dedup_examples(examples)
    return examples[:repo_max], repo_name

extract_jobs = [
    (clone_results[repo['name']][1], repo['category'], repo['name'],
     repo.get('max_per_repo', max_per_repo))
    for repo in CTF_WRITEUP_REPOS
    if clone_results[repo['name']][0]
]

# Bound by CPU count to avoid swap thrash on small machines; min 4 workers.
n_workers = max(4, min(8, os.cpu_count() or 4))

with ProcessPoolExecutor(max_workers=n_workers) as pool:
    futures = {pool.submit(_extract_repo, job): job[2] for job in extract_jobs}
    for fut in concurrent.futures.as_completed(futures):
        examples, repo_name = fut.result()
        all_examples.extend(examples)
        print(f"  ✓ {repo_name}: {len(examples)} examples extracted")
```

### What this changes

- **Time**: 5-15 repositories × ~2 k markdown files × regex. On a 4-core
  machine, ~3-4× faster extraction phase. Expected **8-12 min saved** per
  full build.
- **Memory**: each worker process opens one repo in memory and dies. With
  `--max-per-repo=500` caps the worst case but expect ~2 GB per worker.

### Verification

After the change, run a small `--source writeups --max-per-repo 50` build:

```bash
time uv run src/build_dataset.py --source writeups --max-per-repo 50
```

Run it twice (warm cache). Compare to today's run time. Expect ~50 % savings
on the second-and-later runs (with cache hot).

---

## Issue 6 — 🟢 LOW: Refine `--skip-system-prompt` and document

### What is wrong

The flag name `--skip-system-prompt` reads as "skip writing system prompt to
output," which is correct. But:

- The convention is `--no-<flag>` for opt-out in Python CLIs, not
  `--skip-<flag>`.
- The flag is **opt-in** (correct default is to keep prompts), but the help
  string doesn't make this crystal-clear, and the trade-off (§distribution
  shift / Gemma 4 chat-template interaction) isn't documented anywhere.

### The fix

`src/process_data.py`, change the argparse:

```python
parser.add_argument(
    "--no-system-prompt",
    action="store_true",
    help=(
        "Omit the system role from output 'messages'. "
        "Training keeps the persona context but excludes from loss only if "
        "your trainer uses assistant_only_loss=True. "
        "Default: ON (system prompt written — recommended)."
    ),
)
```

Rename the internal arg handler accordingly. In `AGENTS.md`, add a short
section under "Known Gotchas" about this flag.

### What this changes

- **Clarity** more than performance. But mis-use breaks Gemma 4 chat templates,
  so the doc matters.

---

## Issue 7 — 🟠 QUALITY INSURANCE: Minimal pytest suite

### Why

After the prior round of fixes, two of them ("`assistant_only_loss=True` is a
one-line fix" and `--skip-system-prompt`) turned out to have hidden subtleties
only caught by web research. **Without tests, similar errors will recur
silently.**

### Minimal pytest targets

#### `tests/test_process_data.py`

```python
from src.process_data import convert_alpaca_to_chat, is_ctf_content

def test_convert_alpaca_to_chat_includes_system_by_default():
    out = convert_alpaca_to_chat("Solve picoCTF", "challenge desc", "ret2win", "pwn")
    assert out["messages"][0]["role"] == "system"

def test_skip_system_prompt_omits_role():
    out = convert_alpaca_to_chat(
        "Solve picoCTF", "challenge desc", "ret2win", "pwn",
        skip_system_prompt=True,
    )
    assert len(out["messages"]) == 2
    assert out["messages"][0]["role"] == "user"
    assert out["messages"][1]["role"] == "assistant"

def test_skip_flag_leaves_user_intact():
    out = convert_alpaca_to_chat(
        "Solve picoCTF", "challenge desc", "ret2win", "pwn",
        skip_system_prompt=True,
    )
    user_msg = out["messages"][0]["content"]
    assert "Solve picoCTF" in user_msg
    assert "challenge desc" in user_msg

def test_is_ctf_content_matches_known_categories():
    for word in ["pwn", "rev", "crypto", "web"]:
        assert is_ctf_content(word)

def test_is_ctf_content_rejects_unrelated_term():
    assert not is_ctf_content("kitchen")  # no overlap
    assert not is_ctf_content("apple")    # pwn matches 'pawn'? oh actually not
```

#### `tests/test_loss_masking.py`

Smoke test that the trainer accepts `assistant_only_loss=True` + a
messages-native dataset without raising.

```python
from datasets import Dataset
from trl import SFTConfig

def test_sftconfig_accepts_assistant_only_loss():
    cfg = SFTConfig(output_dir="/tmp/x", assistant_only_loss=True)
    assert cfg.assistant_only_loss is True

def test_messages_dataset_roundtrip():
    ds = Dataset.from_list([{
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "ans"},
        ]
    }])
    assert "messages" in ds.column_names
    assert ds[0]["messages"][-1]["role"] == "assistant"
```

### Verification

```bash
uv pip install pytest --dev
uv run pytest tests/ -v
```

Expect all tests pass.

---

## Issue 8 — 🟢 LOW: De-dup `extract_from_huggingface` (cross-dataset)

### What is wrong

`src/build_dataset.py:188-227` (`extract_from_huggingface`) loads HF datasets
and `src/download_datasets.py` also loads them with similar logic but
different schemas. And `finetune.sh` invokes **both** paths:

```
uv run src/build_dataset.py --source writeups       # ← also fetches HF data
uv run src/download_datasets.py --dataset all       # ← fetches HF data
```

Result: `kyleavery/picoctf` and `justinwangx/CTFtime` are downloaded
**twice**, and concatenated into the merged dataset. Cross-dataset dedup
catches the second copy, but it costs network and disk on every build.

### The fix

`finetune.sh` (or `src/build_dataset.py`) chooses ONE entry point. The cleanest
split is:

- `build_dataset.py` = GitHub repos + local scraping only.
- `download_datasets.py` = HuggingFace only.

Then in `build_writeups_dataset` (build_dataset.py:283-289), drop the
`hf_datasets = [("kyleavery/picoctf", 500), ("justinwangx/CTFtime", 2000)]`
block. Those tracks are entirely the domain of `download_datasets.py`.

### What this changes

- **Wasted time**: ~1-2 min removed per build (second HF download).
- **Cleaner responsibilities**: each script does one thing.

---

## Order of execution

A sensible order, ordered by ROI:

1. **Issue 1, Step A + B** — loss masking. This is *the* unlocker.
2. **Issue 2 + Issue 3** — eval + save limits. One continuous edit.
3. **Issue 4** — shorten system prompts. 5 minutes.
4. **Issue 5** — parallelize extraction. 1 hour.
5. **Issue 7** — pytests for Issues 1, 2, 4.
6. **Issue 6 + Issue 8** — docs and code-cleanup polish.

After 1-3 are landed:

- Test smoke: build with `--max-samples 200`, fine-tune 50 steps with `qwen35-4b`, check that `assistant_only_loss=True` actually produces a non-trivial
  loss on a tiny dataset (verify via `trainer.state.log_history`).
- Optionally compare eval loss with vs. without `assistant_only_loss=True`
  to confirm the improvement.

After 5:

- Run `uv run pytest tests/ -v` — all green.

---

## What this plan does NOT address (deliberately deferred)

These were called out in the original 33-issue review but are out-of-scope for
"unlock monster Qwen 9B CTF" — each is a stylistic / maintainability
improvement rather than a quality blocker.

- Consolidating the 7 near-identical `download_*()` functions into a config
  table (does not affect training quality).
- Type hints throughout `src/` (no runtime impact).
- Logging module replacing `print()` (cosmetic).
- Removing `--max-per-repo 300` magic numbers from `finetune.sh` (functionally
  fine).
- Tests for upstream `build_dataset.py` extraction regex (covered by data
  quality in process + eval loss).

If time allows: the highest-value cleanup among these is consolidating
`download_*()` — cuts ~40 % of the file's LOC and reduces error surface.

---

## Verification matrix (post-implementation)

| What to verify | How | Expected |
|----------------|-----|----------|
| Loss masking works | `trainer.state.log_history` includes `eval_loss` decreasing only on assistant tokens | yes |
| Eval checkpoint kept | `output_dir/checkpoint-*/trainer_state.json` has `best_model_checkpoint` | one of `checkpoint-100/200/...` |
| Save limit honored | Only one checkpoint in `output_dir/` after a 3-epoch run | 1 |
| System prompt shortened | `len(SystemPromptCTF) < 200` | yes |
| Parallel extraction | `time` of small build is reduced | ~50% on warm cache |
| Pytests pass | `uv run pytest tests/ -v` | all green |

---

## Final go/no-go gate

Before kicking off a "production" Qwen 3.5 9B full-mode run on a fresh Colab
T4 session, *all* of Issues 1-3 must be landed. Without loss masking, the
fine-tune will work but the model will spend capacity on memorization rather
than CTF skill acquisition — defeating the "monster" goal.

Once 1-3 are landed, a single short-mode smoke run (500 samples, 1 epoch)
takes ~30 min and will give a clear signal on whether the eval loss curve is
healthy. If yes → schedule the full-mode run. If not → debug before scaling.
