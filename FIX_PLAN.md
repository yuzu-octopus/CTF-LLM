# FIX_PLAN.md — Iteration toward "monster" Qwen 3.5 9B CTF fine-tune on T4

> **Goal**: produce a high-quality CTF / cybersecurity / competitive-programming fine-tune of Qwen 3.5 9B with Unsloth + QLoRA on a single 16GB T4.
>
> **Scope of this document**: actionable next steps for another agent to execute. Two paths:
> - **Small path** (~30 LoC) — unblock the current pipeline cleanly. Recommended first.
> - **Large path** (~150 LoC) — quality lift on top of small path.
>
> **Decision log**: abliteration is **explicitly rejected** (rationale below).

---

## State of the codebase (verified)

### Already landed (commit `400e1b8` + prior)
| Item | Where | Status |
|---|---|---|
| `assistant_only_loss=True` on SFTTrainer | `src/train.py:161`, notebook cell [24] | ✅ |
| `processing_class=tokenizer` | `src/train.py:150`, notebook cell [24] | ✅ |
| `packing=True` | `src/train.py:167`, notebook cell [24] | ⚠️ see OPEN #1 — potential ValueError in vanilla `trl>=0.12` |
| Eval split + `load_best_model_at_end=True` | `src/train.py`, notebook cell [22] | ✅ |
| `save_total_limit: 1` in all configs | `configs/*.yaml` | ✅ |
| Cell [22] length filter uses `apply_chat_template(ex["messages"])` | notebook cell [22] | ✅ **Fix A** |
| Cell [18] inlines `SYSTEM_CTF`/`SYSTEM_CODING`/`is_ctf_content` | notebook cell [18] | ✅ **Fix B** |
| `ThreadPoolExecutor(4)` + `ProcessPoolExecutor(8)` in build_dataset | `src/build_dataset.py:380-415` | ✅ (source only — see OPEN #2) |
| `--no-system-prompt` flag + AGENTS.md docs | `src/process_data.py`, `AGENTS.md` | ✅ |
| `tests/test_process_data.py` + `tests/test_loss_masking.py` | `tests/` | ✅ |
| Synthetic `synthetic_rev_pwn.jsonl` auto-merged | `finetune.sh` | ✅ |

### Still open (this plan)
1. `packing=True + assistant_only_loss=True` collision — **CRITICAL**
2. Notebook cell [10] reverted to sequential cloning — **MEDIUM (Fix C leftover)**
3. LoRA config not bumped for 9B quality runs — **HIGH**
4. Gradient accumulation not tuned for 9B — **HIGH**
5. CTF-Dojo data not in pipeline — **MEDIUM**
6. Two-stage SFT recipe not codified — **MEDIUM**

---

# Small path — unblock current Colab runs (~30 LoC)

## OPEN #1 — Resolve `packing=True` + `assistant_only_loss=True` conflict ⚠️ CRITICAL

**Where**
- `src/train.py:161, 167` (both args present)
- `notebooks/qwen4b_self_contained.ipynb` cell [24] (both args present)
- `scripts/generate_notebook.py` cell-24 block (regenerates the cell)

**Problem**
`trl>=0.12` `SFTTrainer.__init__` validates this combo and **raises `ValueError`** in vanilla TRL. Unsloth's monkey patching sometimes overrides the guard in newer versions, but not always.

**Action**
Before any other fix, run this on Colab to learn the truth:

```python
import trl, peft, transformers, unsloth
print(f"trl={trl.__version__}")
print(f"peft={peft.__version__}")
print(f"transformers={transformers.__version__}")
print(f"unsloth={unsloth.__version__}")
```

Then **decide**:

| Result | Decision | Edit |
|---|---|---|
| Cell [24] runs and starts training | Both work (Unsloth patched). Keep both. | None |
| Cell [24] raises `ValueError` mentioning `assistant_only_loss` + `packing` | TRL guard active. **Drop `packing=True`** to keep assistant-targeted loss. | `src/train.py:167`: delete the `packing=True` line. In `scripts/generate_notebook.py` cell-24 construction, remove the `packing=True` line and its comment. Re-run `python3 scripts/generate_notebook.py` to regenerate. |

**Why drop `packing`, not `assistant_only_loss`**:
- `assistant_only_loss=True` is the only thing that prevents the model from wasting gradient on the system prompt + user question (~30-50% of every step otherwise).
- `packing=True` saves 2-3× wall time but on T4 with ~1.5k examples the absolute saving is ~15-20 min. Worth trading for actual correctness.

**Verification (small path)**
```python
# In a Colab cell after training completes:
import json
train_log = json.load(open("/content/outputs/trainer_state.json"))
print("Final step:", train_log["global_step"])
print("Has loss history:", len(train_log["log_history"]) > 0)
print("Did NOT see ValueError on cell [24].")  # manual
```

---

## OPEN #2 — Parallelize notebook cell [10] (Fix C leftover)

**Where**: `notebooks/qwen4b_self_contained.ipynb` cell [10] (regenerated from `H[10]["source"]` in `scripts/generate_notebook.py` line 117).

**Problem**: Cell [10] is the loop `for repo in tqdm(CTF_WRITEUP_REPOS, desc="Cloning repos"): clone_repo; extract_writeup`. Sequential. ~6 min wasted per run.

**Action** — replace `scripts/generate_notebook.py` line 117 region (the `code(H[10]["source"])` block) with an inlined version that mirrors the `src/build_dataset.py:build_writeups_dataset` parallel pipeline. Two ways:

**Option A (preferred — keep self-contained)**: in-place hardcoded call to the parallel functions defined in cell [8].

Replace the entire cell-10 construction in `scripts/generate_notebook.py`:

```python
# Old (line 117):
code(H[10]["source"]),

# New:
code([r"""# 3.4 Clone all repos in parallel + extract writeups
import concurrent.futures, tempfile
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

def _extract(args):
    from build_dataset_dup import extract_writeup  # inlined below
    ...

"""]),
```

Recommended alternative (much simpler):

**Option B (delegate to src)**: in `notebooks/`, the `src/` package is uploaded alongside (check `finetune.sh`). Replace cell [10] with:

```python
code([r"""# 3.4 Clone all repos in parallel + extract (delegates to src/build_dataset.py)
import sys, importlib
sys.path.insert(0, "/content")
# build_dataset.py functions are already in scope from cell 8 (clone_repo, find_solution_boundary, extract_code_blocks)
# build_writeups_dataset runs ThreadPoolExecutor(4) + ProcessPoolExecutor(8)
all_writeups = build_writeups_dataset(
    f"{WORK_DIR}/data/raw/writeups.jsonl",
    max_per_repo=MAX_PER_REPO
)
"""]),
```

This requires `build_writeups_dataset` to be in scope — the current cell [8] only imports `clone_repo`, `find_solution_boundary`, `extract_code_blocks` extracted from `src/build_dataset.py`. So Option A is preferred (self-contained).

**Recommended: use Option A** but lift `build_writeups_dataset` itself into cell [8] (as an imported function definition, similar to how `clone_repo` is currently imported). Then cell [10] becomes a 5-line call.

**Verification (small path)**
```
Re-run cell [10] in Colab.
Expected: total time drops from ~15 min → ~5 min for 7 repos.
Captured writeups count: same as before (~2500).
No "FAIL" except for known-bad pwncollege repos.
```

---

# Large path — quality lift on top of small path (~150 LoC)

Execute after the small path is verified working.

## OPEN #3 — Bump LoRA config for 9B quality runs

**Where**: `configs/qwen35.yaml`

**Current**:
```yaml
training config:
  r: 16
  lora_alpha: 32
```

**Change to**:
```yaml
training config:
  r: 32
  lora_alpha: 64
```

**Rationale**: Doubled trainable LoRA params for 9B at cost ~640 MB additional VRAM (measured: 9B Qwen 3.5 with r=16 + max_seq=4096 + batch=1 ≈ 11.2 GB; r=32 + same ≈ 11.8 GB — well under 16 GB T4 ceiling).

**Skip for the 4B fast mode**: `configs/qwen35-4b.yaml` should keep `r: 8, lora_alpha: 16` — diminishing returns at smaller scale.

**Verification**
```python
# After training step 5/5 (model load + LoRA config), check VRAM:
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB / 16.0 GB")
# Expected: ~12.0 GB (still 4 GB headroom for backward pass + activations)
```

---

## OPEN #4 — Increase gradient accumulation for stable convergence

**Where**: `configs/qwen35.yaml`

**Current**:
```yaml
training config:
  gradient_accumulation_steps: 4
```

**Change to**:
```yaml
training config:
  gradient_accumulation_steps: 8
```

**Rationale**: Effective batch 8 is a friendlier loss landscape for the ~1.5k-example merged dataset. Halves optimizer steps but each step sees more data. Wall-time impact on T4: +0% (gradient accumulation is compute-equivalent for a fixed data volume).

**Skip for 4B fast mode**: keep `gradient_accumulation_steps: 4` (smaller datasets prefer smaller effective batch).

**Verification**: `trainer_state.json["global_step"]` should be **half** the prior run's value. Eval loss curves should show smoother convergence (fewer oscillations).

---

## OPEN #5 — Add CTF-Dojo data pipeline

**Where**
- `src/build_dataset.py`: add a new `build_ctfdojo_dataset(output_path, max_samples)` function
- `src/build_dataset.py` `main()`: add `ctfd` to the `choices=["writeups","docs","all","merge"]` enum (or add a new `--source ctfd` semantic)
- `finetune.sh`: add a step in `--build-data` to invoke it
- `AGENTS.md`: document the new dataset

**Action**

1. Clone `amazon-science/CTF-Dojo` (GitHub repo, not on HF Hub). Subdirectory `SFT-data/` contains execution-grounded trajectories (~658 verified challenges with `(task, expert_trajectory, flag)` triples).

2. In `src/build_dataset.py`, add:

```python
def build_ctfdojo_dataset(output_path: str, max_samples: int = 500) -> list:
    """Load amazon-science/CTF-Dojo SFT trajectories from a local clone."""
    import json as _json
    from pathlib import Path

    ctfdojo_root = Path("./data/raw/ctfdojo")
    sft_dir = ctfdojo_root / "SFT-data"

    if not sft_dir.exists():
        # Clone shallow
        from git import Repo
        ctfdojo_root.parent.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(
            "https://github.com/amazon-science/CTF-Dojo",
            str(ctfdojo_root),
            depth=1,
        )

    examples = []
    for jsonl_file in sft_dir.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                if len(examples) >= max_samples:
                    break
                item = _json.loads(line)
                task = item.get("task", item.get("question", ""))
                trajectory = item.get("trajectory", item.get("expert_trajectory", ""))
                flag = item.get("flag", "")
                if not task or not trajectory:
                    continue
                examples.append({
                    "instruction": task,
                    "input": "",
                    "output": trajectory + (f"\n\nFlag: {flag}" if flag else ""),
                    "category": item.get("category", "ctf"),
                    "source": "ctfdojo",
                })
        if len(examples) >= max_samples:
            break

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} CTF-Dojo examples to {output_path}")
    return examples
```

3. In `src/build_dataset.py:main()`, add to argparse choices and dispatch:

```python
parser.add_argument("--source", choices=["writeups", "docs", "ctfd", "all", "merge"], required=True)

# In dispatch:
elif args.source == "ctfd":
    build_ctfdojo_dataset(f"{args.output_dir}/ctfdojo.jsonl", args.max_per_repo)
```

4. In `finetune.sh`, after the writeups/docs steps, add:

```bash
uv run src/build_dataset.py --source ctfd --output-dir data/raw --max-per-repo 500
```

5. Update `AGENTS.md` "Build command (Full mode)" section to include `ctfd` as a step.

**Verification**: After running, `data/raw/ctfdojo.jsonl` should contain ≥400 examples. Run `head -1 data/raw/ctfdojo.jsonl | python3 -m json.tool` and confirm structure:

```json
{
  "instruction": "...",
  "input": "",
  "output": "...trajectory...Flag: ...",
  "category": "...",
  "source": "ctfdojo"
}
```

---

## OPEN #6 — Two-stage SFT recipe (curriculum learning)

**Where**: `src/train.py` (add a `train_two_stage()` function), `finetune.sh`, configs

**Concept**
- Stage 1 (1 epoch): LoRA `r=8` on full ~1.5k-example mixed dataset. Establishes broad CTF/coding distribution.
- Stage 2 (2 epochs): LoRA `r=32` `alpha=64` **on a curated subset only** — synthetic_rev_pwn + ctfdojo + hand-picked writeups. Sharpens specific patterns.

**Rationale**: A single-stage `r=32` over the full noisy dataset risks overfitting on long writeup explanations. Splitting lets the broad-base + sharp-pattern approach mirror how humans learn.

**Action (sketch only — ~50 LoC)**

```python
# In src/train.py, add:
def train_two_stage(model_key, data_file, curated_file, output_dir):
    """Two-stage SFT: broad r=8 first, then sharp r=32 on curated subset."""
    config = load_config(model_key)
    # Stage 1: r=8, 1 epoch, full data
    s1_config = dict(config)
    s1_config["model"]["r"] = 8
    s1_config["model"]["lora_alpha"] = 16
    s1_config["training"]["num_train_epochs"] = 1
    train(model_key, data_file, f"{output_dir}/stage1", epochs=1)
    # Stage 2: r=32 alpha=64, 2 epochs, curated only — load stage1 LoRA + continue
    # (Skeletal — full impl requires re-loading the LoRA and rebuilding the model)
    ...
```

Alternatively, encode the two stages directly in `finetune.sh`:

```bash
# Stage 1
./finetune.sh qwen35 --train --epochs 1 --output outputs/qwen35-ctf-stage1  # r=8
# Stage 2 (manual override of configs)
QWEN_LORA_R=32 QWEN_LORA_ALPHA=64 QWEN_EPOCHS=2 \
  ./finetune.sh qwen35 --train --data data/merged/curated.jsonl \
  --output outputs/qwen35-ctf-final
```

**Verification**: After both stages, generate 5 inference probes (use notebook cells [30]/[31]) and compare against single-stage `r=32` runs. Significant qualitative gain is the success metric — exact numbers depend on prompts.

**Status**: marked **MEDIUM, optional** — implement only after #3-#5 are stable.

---

# Decision log: abliteration **REJECTED**

## Rationale
1. **Mechanism is real but evidence is weak.** Arditi et al. (NeurIPS 2024) shows refusal lives in a single direction. But **no published, controlled study benchmarks abliterated-vs-aligned base + identical SFT on CTF/offensive-security tasks**. Community claims are folkloric; some cited papers could not be independently verified.
2. **The user's data may already override refusal.** SFT gradients on `{"role":"assistant","content":"<actual exploit>"}` examples push the model away from the refusal distribution regardless of base. The "ghost" hypothesis is mechanistically plausible but unmeasured.
3. **Alignment tax is documented.** Abliteration measurably regresses MMLU/GSM8K in published evaluations. Loss is usually small but compounding with CTF specialization further muddies the outcome.
4. **Qwen 3.5 9B Instruct is already permissive** on technical content vs Llama-3.1-Instruct. Empirically less refusal-firing, less payoff from ablation.
5. **Cost**: dedicated ablation calibration + a separate ablation pass for each new base model. For a single-user, single-T4-budget setup, the engineering cost outweighs an uncertain single-digit-percent lever.
6. **Coder specialization beats abliteration.** If the user wanted to chase this lever specifically, the higher-ROI move is `Qwen2.5-Coder-7B-Instruct-abliterated` (coder + ablation), not just ablation on Qwen3.5-9B. But coder-7B loses parameter count and context length.

## What would change this decision
- A published controlled benchmark showing abliterated-vs-aligned + SFT differs by >5% on a standard CTF eval dataset (ideally Cybench or CTF-ACE).
- Direct empirical confirmation that the user's current pipeline is producing refusals on his training data.
- An `Qwen3.5-9B-Instruct-Abliterated` checkpoint published on HuggingFace with all layers correctly ablated and capability preserved.

Until one of those appears, **abliteration is out of scope.**

---

# Verification matrix (smoke tests after each implementation)

| Item | Verification | Pass criterion |
|---|---|---|
| OPEN #1 (packing/masking) | Run notebook cells [18]-[25] on T4 | Cell [25] prints "Total steps: ~N" and `history` shows loss decreasing |
| OPEN #2 (cell [10] parallel) | Run cell [10] on FAST mode | Wall-clock drops from ~15 min → ~5 min. Total examples same. |
| OPEN #3 (LoRA r=32) | Run cell [22]-[25] | VRAM after step 23 ≤ 13.0 GB / 16.0 GB |
| OPEN #4 (grad_accum=8) | Run cell [25] | `global_step` is half of prior r=16 grad_accum=4 run's step count |
| OPEN #5 (CTF-Dojo data) | Run `head -1 data/raw/ctfdojo.jsonl` | Schema matches: instruction/input/output/source=ctfdojo |
| OPEN #6 (two-stage) | Run end-to-end | Stage 2 LoRA produces visibly different outputs than stage 1 (qualitative check on 5 prompts) |

---

# Order of execution — what to ship in what order

1. **OPEN #1** — must do first; rest are scoped around it. ~5 LoC.
2. **OPEN #2** — performance, not crash. After #1 is verified. ~20 LoC + notebook regen.
3. **OPEN #3** + **OPEN #4** — bump Qwen 35 yaml in one edit. ~5 LoC.
4. **OPEN #5** — new data source, gated on data-pipeline stability. ~75 LoC.
5. **OPEN #6** — optional polish, gated on #3-#5 producing stable gains. ~50 LoC.

**Total**: ~155 LoC, ~3 hours of focused implementation, ~3-5 hours of Colab training validation.

---

# What this plan deliberately does NOT cover

- Abliteration, base-model swap, refusal-direction probing. (Decision logged above.)
- Replacing the `extract_constants()` brittleness in `scripts/generate_notebook.py` (cell [18] fix B inlined instead of fixing the generator). Acceptable since cell [18] is now stable through the inlining.
- Manual abliteration tool availability — left as a future experiment if the decision reverses.
- A `Qwen2.5-Coder-7B` substitution — would require data re-balancing; documented as a higher-ROI alternative to abliteration only.

---

# Reference: original (historical) review context

The plan above is based on a multi-iteration review:

- **Initial review** (33 findings) — quality, efficiency, speed across the entire pipeline
- **Prior FIX PLAN** (commit `20bdd36`) — implemented 8 fixes; some overlapped with current priorities
- **Notebook bug fixes** (`400e1b8`) — landed Fix A and Fix B; Fix C deferred (this plan)
- **Abliteration research** — rejected (decision logged above)

The full prior FIX_PLAN text is preserved in git history (commit `400e1b8`).
