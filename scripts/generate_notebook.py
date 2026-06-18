#!/usr/bin/env python3
"""Generate notebooks/qwen4b_self_contained.ipynb from src/*.py functions.
Run: python3 scripts/generate_notebook.py
"""
import json
import re
from pathlib import Path

PROJECT = Path(__file__).parent.parent
SRC = PROJECT / "src"
NB_PATH = PROJECT / "notebooks" / "qwen4b_self_contained.ipynb"

SRC_FILES = {
    "build_dataset": SRC / "build_dataset.py",
    "download_datasets": SRC / "download_datasets.py",
    "process_data": SRC / "process_data.py",
}

src_content = {}
for name, path in SRC_FILES.items():
    src_content[name] = path.read_text()

def extract_function(filepath, func_name):
    content = src_content[filepath]
    lines = content.split("\n")
    func_lines = []
    in_func = False
    func_indent = 0
    for line in lines:
        if re.match(rf"^def {func_name}\(", line):
            in_func = True
            func_indent = len(line) - len(line.lstrip())
            func_lines.append(line)
            continue
        if in_func:
            if line.strip() == "":
                func_lines.append(line)
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent > func_indent:
                func_lines.append(line)
            elif re.match(r"^def |^class ", line):
                break
            else:
                func_lines.append(line)
    return "\n".join(func_lines).rstrip()

def extract_constants(filepath, names):
    content = src_content[filepath]
    lines = content.split("\n")
    result = []
    for name in names:
        for i, line in enumerate(lines):
            if line.startswith(f"{name} =") or line.startswith(f"{name} = "):
                if '"""' in line or "\'\'\'" in line:
                    quote = '"""' if '"""' in line else "\'\'\'"
                    collected = [line]
                    for j in range(i+1, len(lines)):
                        collected.append(lines[j])
                        if quote in lines[j] and j > i:
                            break
                    result.append("\n".join(collected))
                    break
                else:
                    result.append(line)
                    break
    return "\n".join(result)

def adapt_for_notebook(source):
    source = source.replace("from tqdm import tqdm", "from tqdm.notebook import tqdm")
    lines = source.split("\n")
    adapted = []
    skip_indent = None
    for line in lines:
        if line.strip().startswith("if HAS_TQDM:") or line.strip().startswith("if not HAS_TQDM:"):
            skip_indent = len(line) - len(line.lstrip())
            continue
        if skip_indent is not None:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() == "" or current_indent > skip_indent:
                continue
            else:
                skip_indent = None
        if "HAS_TQDM" in line and "= True" not in line and "= False" not in line:
            continue
        adapted.append(line)
    source = "\n".join(adapted)
    source = source.replace("[:3000]", "[:MAX_OUTPUT_LEN]")
    source = source.replace("[:2000]", "[:MAX_OUTPUT_LEN]")
    source = source.replace("[:1500]", "[:MAX_OUTPUT_LEN]")
    return source

src_funcs = {
    "clone_repo": adapt_for_notebook(extract_function("build_dataset", "clone_repo")),
    "find_solution_boundary": adapt_for_notebook(extract_function("build_dataset", "find_solution_boundary")),
    "extract_code_blocks": adapt_for_notebook(extract_function("build_dataset", "extract_code_blocks")),
    "load_hf_with_fallback": adapt_for_notebook(extract_function("download_datasets", "load_hf_with_fallback")),
    "extract_qa": adapt_for_notebook(extract_function("download_datasets", "extract_qa")),
    "scrape_doc": adapt_for_notebook(extract_function("build_dataset", "scrape_documentation")),
    "system_prompts": extract_constants("process_data", ["SYSTEM_PROMPT_CTF", "SYSTEM_PROMPT_CODING"]),
    "is_ctf_content": adapt_for_notebook(extract_function("process_data", "is_ctf_content")),
    "ctf_keywords": adapt_for_notebook(extract_constants("process_data", ["CTF_KEYWORDS"])),
}

with open(NB_PATH) as f:
    current_nb = json.load(f)

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
H = current_nb["cells"]

cells = [
    md(H[0]["source"]), md(H[1]["source"]), code(H[2]["source"]), code(H[3]["source"]),
    md(H[4]["source"]), code(H[5]["source"]), md(H[6]["source"]), code(H[7]["source"]),
    # Cell 8: FROM SRC
    code(["# 3.2 Helper functions (from src/build_dataset.py)\n",
          src_funcs["clone_repo"] + "\n\n" + src_funcs["find_solution_boundary"] + "\n\n" + src_funcs["extract_code_blocks"] + "\n\nprint(\"Helper functions defined\")"]),
    code(H[9]["source"]), code(H[10]["source"]), md(H[11]["source"]), code(H[12]["source"]),
    # Cell 13: FROM SRC
    code(["# 4.2 Download and convert HF datasets to Alpaca format\n# (from src/download_datasets.py)\n",
          src_funcs["load_hf_with_fallback"] + "\n\n" + src_funcs["extract_qa"]]),
    md(H[14]["source"]), code(H[15]["source"]),
    # Cell 16: FROM SRC
    code(["# 5.2 Scrape and parse documentation\n# (from src/build_dataset.py)\n",
          src_funcs["scrape_doc"]]),
    md(H[17]["source"]),
    # Cell 18: FROM SRC
    code(["# 6.1 System prompts (from src/process_data.py)\n",
          src_funcs["system_prompts"] + "\n\n" + src_funcs["ctf_keywords"] + "\n\n" + src_funcs["is_ctf_content"] + "\n\nprint(\"System prompts defined\")"]),
    code(H[19]["source"]), md(H[20]["source"]), code(H[21]["source"]),
    # Cell 22: Dataset loading + eval split
    code(["# 7.4 Load dataset and apply chat template\n",
          "dataset = load_dataset(\"json\", data_files={\"train\": f\"{WORK_DIR}/data/merged/train.jsonl\"}, split=\"train\")\n",
          "print(f\"Dataset (raw): {len(dataset)} examples\")\n",
          "\n",
          "# Drop empty assistant outputs\n",
          "def has_assistant(ex):\n",
          "    msgs = ex.get(\"messages\", [])\n",
          "    if not msgs or not msgs[-1].get(\"content\"):\n",
          "        return False\n",
          "    return True\n",
          "dataset = dataset.filter(has_assistant, desc=\"Filter empty outputs\")\n",
          "print(f\"Dataset (no empty outputs): {len(dataset)} examples\")\n",
          "\n",
          "# Length filter: drop samples longer than max_seq_length tokens (speed win)\n",
          "# Note: Qwen 3.5 returns a Qwen3VLProcessor (multimodal) that wraps a tokenizer.\n",
          "# We need the inner tokenizer for encode().\n",
          "actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer\n",
          "print(f\"Tokenizer type: {type(tokenizer).__name__}\")\n",
          "def length_ok(ex):\n",
          "    tokens = actual_tokenizer.encode(ex[\"text\"], add_special_tokens=False)\n",
          "    return len(tokens) <= MAX_SEQ_LENGTH\n",
          "\n",
          "before = len(dataset)\n",
          "dataset = dataset.filter(length_ok, desc=\"Filter by length\")\n",
          "after = len(dataset)\n",
          "if before != after:\n",
          "    print(f\"Dataset (length-filtered): {after} examples (dropped {before - after} long samples)\")\n",
          "else:\n",
          "    print(f\"Dataset (length-filtered): {after} examples (all within {MAX_SEQ_LENGTH} tokens)\")\n",
          "\n",
          "print(f\"\\nSample:\\n{dataset[0]['text'][:300]}...\")\n",
          "\n",
          "# Split 10% for eval to detect overfitting\n",
          "split = dataset.train_test_split(test_size=0.1, seed=42)\n",
          "train_dataset = split['train']\n",
          "eval_dataset = split['test']\n",
          "print(f\"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples\")"]),
    md(H[23]["source"]),
    # Cell 24: SFTTrainer with eval strategy
    code(["# ============================================================\n",
          "# 8.1 Create custom training callback with rich visual indicators\n",
          "# ============================================================\n",
          "from trl import SFTTrainer, SFTConfig\n",
          "from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl\n",
          "\n",
          "class RichTrainingCallback(TrainerCallback):\n",
          '    """Custom callback with tqdm progress bar, ETA, GPU memory, and detailed metrics."""\n',
          "\n",
          "    def __init__(self):\n",
          "        self.progress_bar = None\n",
          "        self.start_time = None\n",
          "        self.loss_history = []\n",
          "        self.lr_history = []\n",
          "\n",
          "    def on_train_begin(self, args, state, control, **kwargs):\n",
          "        self.start_time = time.time()\n",
          "        self.progress_bar = tqdm(\n",
          "            total=state.max_steps,\n",
          '            desc="Training",\n',
          '            unit="step",\n',
          '            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"\n',
          "        )\n",
          "        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0\n",
          '        print(f"\\n{\'=\'*70}")\n',
          '        print(f"  TRAINING STARTED")\n',
          '        print(f"  Total steps: {state.max_steps} | Epochs: {NUM_EPOCHS}")\n',
          '        print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")\n',
          '        print(f"  Learning rate: {args.learning_rate:.2e}")\n',
          '        print(f"  VRAM allocated: {gpu_mem:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")\n',
          '        print(f"{\'=\'*70}\\n")\n',
          "\n",
          "    def on_log(self, args, state, control, logs=None, **kwargs):\n",
          "        if logs is None or self.progress_bar is None:\n",
          "            return\n",
          '        loss = logs.get("loss", 0)\n',
          '        lr = logs.get("learning_rate", 0)\n',
          '        epoch = logs.get("epoch", 0)\n',
          "\n",
          "        self.loss_history.append(loss)\n",
          "        self.lr_history.append(lr)\n",
          "\n",
          "        # GPU memory\n",
          "        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0\n",
          "\n",
          "        # Time estimates\n",
          "        elapsed = time.time() - self.start_time\n",
          "        step_time = elapsed / max(state.global_step, 1)\n",
          "        eta_sec = step_time * (state.max_steps - state.global_step)\n",
          "\n",
          "        # Update progress bar postfix\n",
          '        self.progress_bar.set_postfix_str(\n',
          '            f"loss={loss:.4f} | lr={lr:.2e} | ep={epoch:.2f} | "\n',
          '            f"vram={gpu_mem:.1f}GB | {step_time:.1f}s/step | eta={eta_sec/60:.1f}min"\n',
          "        )\n",
          "\n",
          "    def on_train_end(self, args, state, control, **kwargs):\n",
          "        if self.progress_bar:\n",
          "            self.progress_bar.close()\n",
          "        total_time = time.time() - self.start_time\n",
          "        final_loss = self.loss_history[-1] if self.loss_history else 0\n",
          '        print(f"\\n{\'=\'*70}")\n',
          '        print(f"  TRAINING COMPLETE")\n',
          '        print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")\n',
          '        print(f"  Final loss: {final_loss:.4f}")\n',
          '        print(f"  Loss reduction: {self.loss_history[0]:.4f} -> {final_loss:.4f} "\n',
          '              f"({(1 - final_loss/self.loss_history[0])*100:.1f}%)" if self.loss_history else "")\n',
          '        print(f"{\'=\'*70}\\n")\n',
          "\n",
          'print("Custom training callback defined")\n',
          'print(f"  - tqdm progress bar with ETA")\n',
          'print(f"  - Loss + learning rate per step")\n',
          'print(f"  - GPU memory monitoring")\n',
          'print(f"  - Epoch tracking")\n',
          'print(f"  - Time estimates")\n',
          "\n",
          "# ============================================================\n",
          "# 8.2 Create SFTTrainer with callback\n",
          "# ============================================================\n",
          "rich_callback = RichTrainingCallback()\n",
          "\n",
          "trainer = SFTTrainer(\n",
          "    model=model,\n",
          "    processing_class=tokenizer,\n",
          "    train_dataset=train_dataset,\n",
          "    eval_dataset=eval_dataset,\n",
          "    args=SFTConfig(\n",
          "        max_seq_length=MAX_SEQ_LENGTH,\n",
          "        per_device_train_batch_size=BATCH_SIZE,\n",
          "        gradient_accumulation_steps=GRAD_ACCUM,\n",
          "        warmup_ratio=WARMUP_RATIO,\n",
          "        num_train_epochs=NUM_EPOCHS,\n",
          "        learning_rate=float(LEARNING_RATE),\n",
          "        weight_decay=WEIGHT_DECAY,\n",
          "        max_grad_norm=MAX_GRAD_NORM,\n",
          "        lr_scheduler_type=LR_SCHEDULER,\n",
          "        logging_steps=10,\n",
          '        output_dir=f"{WORK_DIR}/outputs",\n',
          '        optim="adamw_8bit",\n',
          "        seed=3407,\n",
          '        save_strategy="epoch",\n',
          "        save_total_limit=1,\n",
          "        fp16=not torch.cuda.is_bf16_supported(),\n",
          "        bf16=torch.cuda.is_bf16_supported(),\n",
          "        assistant_only_loss=True,\n",
          "        dataset_num_proc=1,\n",
          '        report_to="none",\n',
          "        # Speed: pack short samples into one sequence (biggest win after max_seq_length)\n",
          "        packing=True,\n",
          "        # Quality: NEFTune adds embedding noise (free at runtime)\n",
          "        neftune_noise_alpha=NEFTUNE_NOISE_ALPHA if NEFTUNE_NOISE_ALPHA else None,\n",
          '        eval_strategy="steps",\n',
          "        eval_steps=100,\n",
          "        load_best_model_at_end=True,\n",
          '        metric_for_best_model="eval_loss",\n',
          "        # rsLoRA: alpha/sqrt(r) scaling (no runtime cost)\n",
          "        # use_rslora configured via get_peft_model in Section 7.3\n",
          "    ),\n",
          "    callbacks=[rich_callback],\n",
          ")\n",
          'print("Trainer created")\n',
          'print(f"  Mode: {MODE.upper()}")\n',
          'print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, Grad accum: {GRAD_ACCUM}")\n',
          'print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")\n',
          'print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout=0")\n',
          'print(f"  LR: {LEARNING_RATE}, scheduler: {LR_SCHEDULER}, warmup: {WARMUP_RATIO}")\n',
          'print(f"  Max seq: {MAX_SEQ_LENGTH}, packing: True, NEFTune: {NEFTUNE_NOISE_ALPHA}, rsLoRA: {USE_RSLORA}")\n',
          'print(f"  Total steps: ~{len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRAD_ACCUM)} (unpacked; will be ~3-5x less with packing)")']),
    code(H[25]["source"]), md(H[26]["source"]),
    code(H[27]["source"]), code(H[28]["source"]), md(H[29]["source"]), code(H[30]["source"]), code(H[31]["source"]),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 4
}

NB_PATH.write_text(json.dumps(notebook, indent=1))
md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Generated {NB_PATH} ({len(cells)} cells: {md_count} markdown, {code_count} code)")
