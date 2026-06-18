# Notebook dump: qwen4b_self_contained.ipynb (32 cells)


## [0] markdown (336 chars)

```
# Fine-tune LLMs for CTF/Coding - Self-Contained Colab Pipeline

Complete pipeline: **data scraping -> synthesis -> training -> export**

**Setup:**
1. Runtime -> Change runtime type -> **T4 GPU**
2. Run cells top-to-bottom (1-11)
3. No external files needed - everything is built in

**Expected runtime:** ~2-3 hours for 3 epochs on T4
```


## [1] markdown (34 chars)

```
## Section 1: Install Dependencies
```


## [2] code (1309 chars)

```
# ============================================================
# 1.1 Detect environment + install packages
# ============================================================
import os, re, sys, json, time, shutil, tempfile
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from pathlib import Path

IS_KAGGLE = "KAGGLE_USERNAME" in os.environ
IS_COLAB = "COLAB_" in "".join(os.environ.keys())
WORK_DIR = "/kaggle/working" if IS_KAGGLE else "/content"
print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'}")

if IS_COLAB or IS_KAGGLE:
    import torch
    v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install -q sentencepiece protobuf datasets huggingface_hub hf_transfer requests gitpython pyyaml tqdm
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"
    !pip install --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"
    # Speed up Qwen3.5 attention path (~2x faster training)
    !pip install --no-deps -q flash-linear-attention causal-conv1d 2>&1 | tail -3

    torch._dynamo.config.recompile_limit = 64
```


## [3] code (1140 chars)

```
# ============================================================
# 1.2 Import libraries and verify environment
# ============================================================
import requests, yaml
from git import Repo
from datasets import load_dataset
from tqdm.notebook import tqdm  # Colab-friendly progress bars
import torch
from IPython.display import display, HTML

print(f"\n=== Environment ===")
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU:      {gpu_name} ({gpu_mem:.1f} GB)")

    if "T4" not in gpu_name:
        print(f"\nWARNING: This pipeline is optimized for T4 (16GB).")
        print(f"   You have {gpu_name} ({gpu_mem:.1f} GB).")

    print(f"\n=== GPU Details ===")
    !nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null || echo "(nvidia-smi not available)"
else:
    print("\nERROR: No GPU detected!")
    print("   Go to Runtime -> Change runtime type -> T4 GPU")
```


## [4] markdown (130 chars)

```
## Section 2: Configuration

All tunable parameters in one place. Edit this cell to adjust model, LoRA rank, training epochs, etc.
```


## [5] code (3434 chars)

```
# ============================================================
# 2.0 Training mode (FAST for iteration, QUALITY for final)
# ============================================================
# FAST: ~30 min on T4. For rapid iteration on data/prompts.
#   - 1 epoch, r=8, max_seq=2048
# QUALITY: ~50-70 min on T4. For final model.
#   - 2 epochs, r=16, max_seq=4096, eval + early stopping
MODE = "fast"  # or "quality"

# ============================================================
# 2.1 Model selection
# ============================================================
# Fast: 4B for iteration, Full: 9B for quality
if MODE == "fast":
    MODEL_NAME = "unsloth/Qwen3.5-4B"   # 3.5GB weights, fast iteration
else:
    MODEL_NAME = "unsloth/Qwen3.5-9B"   # 6GB weights, better quality

# ============================================================
# 2.2 LoRA hyperparameters (model-dependent)
# ============================================================
if "4B" in MODEL_NAME:
    LORA_R = 8
    LORA_ALPHA = 16      # alpha/r = 2
else:  # 9B
    LORA_R = 16
    LORA_ALPHA = 32      # alpha/r = 2
MAX_SEQ_LENGTH = 2048 if MODE == "fast" else 4096

# ============================================================
# 2.3 Training hyperparameters (mode-dependent)
# ============================================================
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4      # lowered from 2e-4 (better for small datasets)
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05 if MODE == "quality" else 0.03
NEFTUNE_NOISE_ALPHA = 5    # free quality boost on small datasets
USE_RSLORA = True if MODE == "quality" else False  # free, no runtime cost
if MODE == "fast":
    NUM_EPOCHS = 1
else:
    NUM_EPOCHS = 2

# ============================================================
# 2.4 Data limits
# ============================================================
if MODE == "fast":
    MAX_PER_REPO = 30       # only high-signal repos
    MAX_HF_SAMPLES = 200   # fewer HF examples
    MAX_DOC_SECTIONS = 5   # minimal docs
    MAX_OUTPUT_LEN = 2000  # tighter content cap
else:  # full/quality
    MAX_PER_REPO = 999999  # no limit — take everything
    MAX_HF_SAMPLES = 999999  # no limit
    MAX_DOC_SECTIONS = 999999  # no limit
    MAX_OUTPUT_LEN = 20000  # generous safety cap
USE_SYNTHETIC = True  # Include 24 synthetic rev/pwn examples

# ============================================================
# 2.5 Output
# ============================================================
OUTPUT_DIR = f"{WORK_DIR}/outputs/qwen4b-ctf"

CONFIG = {
    "mode": MODE,
    "model_name": MODEL_NAME,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "max_seq_length": MAX_SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_grad_norm": MAX_GRAD_NORM,
    "lr_scheduler": LR_SCHEDULER,
    "warmup_ratio": WARMUP_RATIO,
    "neftune_noise_alpha": NEFTUNE_NOISE_ALPHA,
    "use_rslora": USE_RSLORA,
    "max_per_repo": MAX_PER_REPO,
    "max_hf_samples": MAX_HF_SAMPLES,
    "output_dir": OUTPUT_DIR,
}
print(f"Configuration (mode={MODE}):")
for k, v in CONFIG.items():
    print(f"  {k:20s} = {v}")
print(f"\n  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
est_time = "~30 min (FAST)" if MODE == "fast" else "~50-70 min (QUALITY)"
print(f"  Estimated training time on T4: {est_time}")
```


## [6] markdown (257 chars)

```
## Section 3: Scrape CTF Writeups from GitHub

Clones 10 GitHub repos and extracts Q&A pairs from writeup markdown files.
Note: 3 pwncollege repos are commented out below (extract 0 examples due to non-standard markdown). Uncomment for Qwen 3.5 9B full run.
```


## [7] code (1368 chars)

```
# 3.1 GitHub repos to scrape
CTF_WRITEUP_REPOS = [
    {"url": "https://github.com/Cajac/picoCTF-Writeups", "name": "picoctf-cajac", "category": "picoctf"},
    {"url": "https://github.com/vivian-dai/PicoCTF2021-Writeup", "name": "picoctf-2021", "category": "picoctf"},
    {"url": "https://github.com/DarkCodeOrg/CryptoHack", "name": "cryptohack-darkcode", "category": "cryptohack"},
    {"url": "https://github.com/AyushSingh-c/Cryptohack", "name": "cryptohack-ayush", "category": "cryptohack"},
    # pwncollege repos commented out: extract 0 examples (non-standard markdown)
    # Uncomment for Qwen 3.5 9B full run with larger data budget
    # {"url": "https://github.com/H3xKatana/pwncollege-writeups", "name": "pwncollege-h3x", "category": "pwncollege"},
    # {"url": "https://github.com/prettyb0iisam/pwncollege-writeups", "name": "pwncollege-prettyb0i", "category": "pwncollege"},
    # {"url": "https://github.com/id-none/pwncollege_writeup", "name": "pwncollege-idnone", "category": "pwncollege"},
    {"url": "https://github.com/Adamkadaban/CTFs", "name": "ctfs-adamkadaban", "category": "multi"},
    {"url": "https://github.com/Cryptogenic/Exploit-Writeups", "name": "exploit-writeups", "category": "pwn"},
    {"url": "https://github.com/ffffffff0x/1earn", "name": "0x1earn", "category": "multi"},
]
print(f"Configured {len(CTF_WRITEUP_REPOS)} repos")
```


## [8] code (1618 chars)

```
# 3.2 Helper functions (from src/build_dataset.py)
def clone_repo(url: str, dest: str) -> bool:
    """Clone a git repo using gitpython"""
    import shutil
    
    # Clean up existing directory if present
    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    
    try:
        Repo.clone_from(url, dest, depth=1)
        return Path(dest).exists()
    except Exception as e:
        print(f"  Failed to clone {url}: {e}")
        return False

def find_solution_boundary(content: str) -> int:
    """Find where the solution/writeup section starts"""
    solution_markers = [
        r'^##\s+(?:Solution|Writeup|Exploit|Answer|Flag|Solution:)',
        r'^###\s+(?:Solution|Writeup|Exploit|Answer|Flag|Solution:)',
        r'^##\s+(?:Solving|Solved|My Solution)',
        r'^###\s+(?:Solving|Solved|My Solution)',
        r'^\*\*(?:Solution|Exploit|Answer|Flag)\*\*',
        r'^>\s*(?:Solution|Exploit|Answer|Flag)',
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        for marker in solution_markers:
            if re.match(marker, line, re.IGNORECASE | re.MULTILINE):
                return i
    return len(lines)

def extract_code_blocks(content: str) -> list:
    """Extract code blocks from markdown content"""
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    code_blocks = []
    for lang, code in matches:
        code = code.strip()
        if len(code) > 10:
            code_blocks.append({"lang": lang or "text", "code": code})
    return code_blocks

print("Helper functions defined")
```


## [9] code (2006 chars)

```
# 3.3 Extract Q&A from writeups
def extract_writeup(md_file, category):
    \"\"\"Convert one markdown file into a Q&A example.\"\"\"
    try:
        content = md_file.read_text(errors='ignore')
        if len(content) < 100:
            return None
        if md_file.name.lower() == "readme.md":
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


## [10] code (1023 chars)

```
# 3.4 Clone all repos and extract writeups
t0 = time.time()
all_writeups = []
with tempfile.TemporaryDirectory() as tmpdir:
    for repo in tqdm(CTF_WRITEUP_REPOS, desc="Cloning repos"):
        dest = f"{tmpdir}/{repo['name']}"
        if clone_repo(repo["url"], dest):
            repo_path = Path(dest)
            count = 0
            md_files = [m for m in list(repo_path.rglob("*.md")) + list(repo_path.rglob("*.MD"))
                       if m.name.lower() != "readme.md"]
            for md in tqdm(md_files, desc=f"  {repo['name']}", leave=False):
                ex = extract_writeup(md, repo["category"])
                if ex:
                    all_writeups.append(ex)
                    count += 1
                    if count >= MAX_PER_REPO:
                        break
            print(f"  OK {repo['name']}: {count} examples")
        else:
            print(f"  FAIL {repo['name']}: SKIPPED")

    elapsed = time.time() - t0
    print(f"\nExtracted {len(all_writeups)} writeups in {elapsed:.1f}s")
```


## [11] markdown (159 chars)

```
## Section 4: Download HuggingFace Datasets

Pulls pre-curated datasets: CTFtime writeups, OpenCodeReasoning (competitive programming), cybersecurity Q&A, etc.
```


## [12] code (582 chars)

```
# 4.1 Define HF datasets to download
HF_DATASETS = [
    ("Jacqkues/ctf_webserver_v0.1", 340),
    ("kyleavery/picoctf", 120),
    ("justinwangx/CTFtime", MAX_HF_SAMPLES),
    ("nvidia/OpenCodeReasoning", 2000 if MODE == "full" else MAX_HF_SAMPLES),
    ("AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1", MAX_HF_SAMPLES),
    # Generic vulnerability dataset commented out: less CTF-specific
    # Uncomment for Qwen 3.5 9B full run with larger data budget
    # ("ayshajavd/code-security-vulnerability-dataset", MAX_HF_SAMPLES),
]
print(f"Configured {len(HF_DATASETS)} HF datasets")
```


## [13] code (972 chars)

```
# 4.2 Download and convert HF datasets to Alpaca format
# (from src/download_datasets.py)
def load_hf_with_fallback(name):
    """Try train split, then test, then default."""
    for split in ["train", "test"]:
        try:
            return load_dataset(name, split=split)
        except Exception:
            continue
    return load_dataset(name)

def extract_qa(item):
    """Extract question/answer from various HF dataset schemas."""
    msgs = item.get("messages")
    if msgs and len(msgs) >= 2:
        q = msgs[0].get("content", "")
        a = msgs[-1].get("content", "")
        if q and a:
            return q, a
    q = item.get("question") or item.get("problem") or item.get("description") or ""
    a = item.get("answer") or item.get("solution") or item.get("flag") or ""
    if q and a:
        return q, a
    text = item.get("text_chunk", "")
    if text:
        return "Explain this CTF writeup and provide the solution", text
    return None, None
```


## [14] markdown (157 chars)

```
## Section 5: Scrape Security Tool Documentation

Pulls READMEs and reference docs for pwntools, angr, z3, pwndbg, ROPgadget, Ropper, sympy, gmpy2, sagemath.
```


## [15] code (1113 chars)

```
# 5.1 Documentation sources
DOC_SOURCES = [
    {"url": "https://raw.githubusercontent.com/Gallopsled/pwntools-write-ups/master/README.md", "name": "pwntools"},
    {"url": "https://raw.githubusercontent.com/angr/angr/master/README.md", "name": "angr"},
    {"url": "https://raw.githubusercontent.com/Z3Prover/z3/master/README.md", "name": "z3"},
    {"url": "https://raw.githubusercontent.com/pwndbg/pwndbg/dev/FEATURES.md", "name": "pwndbg"},
    {"url": "https://raw.githubusercontent.com/JonathanSalwan/ROPgadget/master/README.md", "name": "ropgadget"},
    {"url": "https://raw.githubusercontent.com/sashs/Ropper/master/README.md", "name": "ropper"},
    {"url": "https://raw.githubusercontent.com/david942j/one_gadget/master/README.md", "name": "one_gadget"},
    {"url": "https://raw.githubusercontent.com/sympy/sympy/master/README.rst", "name": "sympy"},
    {"url": "https://raw.githubusercontent.com/gmpy2/gmpy2/master/README.md", "name": "gmpy2"},
    {"url": "https://raw.githubusercontent.com/sagemath/sage/master/README.md", "name": "sagemath"},
]
print(f"Configured {len(DOC_SOURCES)} doc sources")
```


## [16] code (1080 chars)

```
# 5.2 Scrape and parse documentation
# (from src/build_dataset.py)
def scrape_documentation(url: str, name: str) -> list:
    """Scrape a single documentation file using requests"""
    examples = []
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        if len(content) < 100:
            return examples
        
        sections = re.split(r'\n(?=## )', content)
        
        for section in sections:
            if len(section) < 50:
                continue
            
            title_match = re.match(r'##\s+(.+)', section)
            title = title_match.group(1) if title_match else "Documentation"
            
            examples.append({
                "instruction": f"Explain how to use {name} for: {title}",
                "input": section[:MAX_OUTPUT_LEN],
                "output": f"Based on {name} documentation:\n\n{section[:MAX_OUTPUT_LEN]}"
            })
    except Exception as e:
        print(f"  Failed to scrape {url}: {e}")
    
    return examples
```


## [17] markdown (156 chars)

```
## Section 6: Convert to ChatML and Merge

Combines all sources (writeups + HF + docs) into ChatML format with auto-selected system prompts (CTF vs coding).
```


## [18] code (352 chars)

```
# 6.1 System prompts (from src/process_data.py)
SYSTEM_PROMPT_CTF = (
SYSTEM_PROMPT_CODING = (

CTF_KEYWORDS = ["pwn", "rev", "web", "crypto", "ctf", "exploit", "vuln", "shellcode"]

def is_ctf_content(text):
    """Check if text contains CTF-related keywords."""
    return any(k in text.lower() for k in CTF_KEYWORDS)

print("System prompts defined")
```


## [19] code (1717 chars)

```
# 6.2 Convert Alpaca -> ChatML format
import sys
sys.path.insert(0, "/content")
try:
    from src.synthetic_rev_pwn import SYNTHETIC_EXAMPLES
except ImportError:
    SYNTHETIC_EXAMPLES = []
    print("Warning: synthetic rev/pwn data not found, skipping")

# Add synthetic rev/pwn examples if enabled
if USE_SYNTHETIC and SYNTHETIC_EXAMPLES:
    synthetic_raw = [{"instruction": e["instruction"], "input": e.get("input",""), "output": e["output"]} for e in SYNTHETIC_EXAMPLES]
    all_writeups += synthetic_raw
    print(f"Added {len(SYNTHETIC_EXAMPLES)} synthetic rev/pwn examples")

all_raw = all_writeups + hf_examples + doc_examples
print(f"Total raw examples: {len(all_raw)}")

chatml_data = []
for ex in tqdm(all_raw, desc="Converting to ChatML"):
    instr = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    if not instr or not out:
        continue
    user_content = f"{instr}\n\n{inp}" if inp else instr
    sys_prompt = SYSTEM_CTF if is_ctf_content(instr) or is_ctf_content(out) else SYSTEM_CODING
    chatml_data.append({
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content[:MAX_OUTPUT_LEN]},
            {"role": "assistant", "content": out[:MAX_OUTPUT_LEN]}
        ]
    })

print(f"\nGenerated {len(chatml_data)} ChatML examples")

# 6.3 Save merged dataset
os.makedirs(f"{WORK_DIR}/data/merged", exist_ok=True)
with open(f"{WORK_DIR}/data/merged/train.jsonl", "w") as f:
    for ex in chatml_data:
        f.write(json.dumps(ex) + "\n")
print(f"Saved to /content/data/merged/train.jsonl")
print(f"  File size: {os.path.getsize(f'{WORK_DIR}/data/merged/train.jsonl') / 1e6:.1f} MB")
```


## [20] markdown (127 chars)

```
## Section 7: Load Model and Configure LoRA

Loads the model with 4-bit QLoRA, applies chat template, configures LoRA adapters.
```


## [21] code (996 chars)

```
# 7.1 Load base model with 4-bit QLoRA
from unsloth import FastLanguageModel, get_chat_template

t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)
print(f"Model loaded: {MODEL_NAME}")
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Load time: {time.time() - t0:.1f}s")

# 7.2 Apply chatml template
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
print("Chat template: chatml")

# 7.3 Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=MAX_SEQ_LENGTH,
    use_rslora=USE_RSLORA,
)
print(f"LoRA configured (r={LORA_R}, alpha={LORA_ALPHA})")
```


## [22] code (1602 chars)

```
# 7.4 Load dataset and apply chat template
dataset = load_dataset("json", data_files={"train": f"{WORK_DIR}/data/merged/train.jsonl"}, split="train")
print(f"Dataset (raw): {len(dataset)} examples")

# Drop empty assistant outputs
def has_assistant(ex):
    msgs = ex.get("messages", [])
    if not msgs or not msgs[-1].get("content"):
        return False
    return True
dataset = dataset.filter(has_assistant, desc="Filter empty outputs")
print(f"Dataset (no empty outputs): {len(dataset)} examples")

# Length filter: drop samples longer than max_seq_length tokens (speed win)
# Note: Qwen 3.5 returns a Qwen3VLProcessor (multimodal) that wraps a tokenizer.
# We need the inner tokenizer for encode().
actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
print(f"Tokenizer type: {type(tokenizer).__name__}")
def length_ok(ex):
    tokens = actual_tokenizer.encode(ex["text"], add_special_tokens=False)
    return len(tokens) <= MAX_SEQ_LENGTH

before = len(dataset)
dataset = dataset.filter(length_ok, desc="Filter by length")
after = len(dataset)
if before != after:
    print(f"Dataset (length-filtered): {after} examples (dropped {before - after} long samples)")
else:
    print(f"Dataset (length-filtered): {after} examples (all within {MAX_SEQ_LENGTH} tokens)")

print(f"\nSample:\n{dataset[0]['text'][:300]}...")

# Split 10% for eval to detect overfitting
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']
print(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")
```


## [23] markdown (445 chars)

```
## Section 8: Train

This is the longest section. In FAST mode (~30 min on T4) you'll see live progress with ETA, loss curves, and per-step metrics. In QUALITY mode (~50-70 min on T4) the same applies but for more steps.

You'll see:
- Live progress bar with **ETA**
- **Loss** and **learning rate** per step
- **GPU memory** monitoring
- **Epoch** progress

If you get disconnected, re-run this cell - it will resume from the latest checkpoint.
```


## [24] code (5461 chars)

```
# ============================================================
# 8.1 Create custom training callback with rich visual indicators
# ============================================================
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class RichTrainingCallback(TrainerCallback):
    """Custom callback with tqdm progress bar, ETA, GPU memory, and detailed metrics."""

    def __init__(self):
        self.progress_bar = None
        self.start_time = None
        self.loss_history = []
        self.lr_history = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.progress_bar = tqdm(
            total=state.max_steps,
            desc="Training",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"\n{'='*70}")
        print(f"  TRAINING STARTED")
        print(f"  Total steps: {state.max_steps} | Epochs: {NUM_EPOCHS}")
        print(f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate:.2e}")
        print(f"  VRAM allocated: {gpu_mem:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        print(f"{'='*70}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.progress_bar is None:
            return
        loss = logs.get("loss", 0)
        lr = logs.get("learning_rate", 0)
        epoch = logs.get("epoch", 0)

        self.loss_history.append(loss)
        self.lr_history.append(lr)

        # GPU memory
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Time estimates
        elapsed = time.time() - self.start_time
        step_time = elapsed / max(state.global_step, 1)
        eta_sec = step_time * (state.max_steps - state.global_step)

        # Update progress bar postfix
        self.progress_bar.set_postfix_str(
            f"loss={loss:.4f} | lr={lr:.2e} | ep={epoch:.2f} | "
            f"vram={gpu_mem:.1f}GB | {step_time:.1f}s/step | eta={eta_sec/60:.1f}min"
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar:
            self.progress_bar.close()
        total_time = time.time() - self.start_time
        final_loss = self.loss_history[-1] if self.loss_history else 0
        print(f"\n{'='*70}")
        print(f"  TRAINING COMPLETE")
        print(f"  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Loss reduction: {self.loss_history[0]:.4f} -> {final_loss:.4f} "
              f"({(1 - final_loss/self.loss_history[0])*100:.1f}%)" if self.loss_history else "")
        print(f"{'='*70}\n")

print("Custom training callback defined")
print(f"  - tqdm progress bar with ETA")
print(f"  - Loss + learning rate per step")
print(f"  - GPU memory monitoring")
print(f"  - Epoch tracking")
print(f"  - Time estimates")

# ============================================================
# 8.2 Create SFTTrainer with callback
# ============================================================
rich_callback = RichTrainingCallback()

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=float(LEARNING_RATE),
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=10,
        output_dir=f"{WORK_DIR}/outputs",
        optim="adamw_8bit",
        seed=3407,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        assistant_only_loss=True,
        dataset_num_proc=1,
        report_to="none",
        # Speed: pack short samples into one sequence (biggest win after max_seq_length)
        packing=True,
        # Quality: NEFTune adds embedding noise (free at runtime)
        neftune_noise_alpha=NEFTUNE_NOISE_ALPHA if NEFTUNE_NOISE_ALPHA else None,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # rsLoRA: alpha/sqrt(r) scaling (no runtime cost)
        # use_rslora configured via get_peft_model in Section 7.3
    ),
    callbacks=[rich_callback],
)
print("Trainer created")
print(f"  Mode: {MODE.upper()}")
print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, Grad accum: {GRAD_ACCUM}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout=0")
print(f"  LR: {LEARNING_RATE}, scheduler: {LR_SCHEDULER}, warmup: {WARMUP_RATIO}")
print(f"  Max seq: {MAX_SEQ_LENGTH}, packing: True, NEFTune: {NEFTUNE_NOISE_ALPHA}, rsLoRA: {USE_RSLORA}")
print(f"  Total steps: ~{len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRAD_ACCUM)} (unpacked; will be ~3-5x less with packing)")
```


## [25] code (1725 chars)

```
# ============================================================
# 8.3 Run training with rich progress visualization
# ============================================================
print("Starting training...")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {len(dataset)} examples")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Watch the progress bar below for real-time updates\n")

t0 = time.time()
trainer.train()
total_time = time.time() - t0
print(f"\nTraining complete in {total_time/60:.1f} minutes")

# ============================================================
# 8.4 Visualize training loss
# ============================================================
import matplotlib.pyplot as plt

if rich_callback.loss_history:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(rich_callback.loss_history, color='#2E86AB', linewidth=2)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Logging Step')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(range(len(rich_callback.loss_history)),
                          rich_callback.loss_history, alpha=0.3, color='#2E86AB')

    # Learning rate curve
    axes[1].plot(rich_callback.lr_history, color='#A23B72', linewidth=2)
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Logging Step')
    axes[1].set_ylabel('Learning Rate')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{WORK_DIR}/training_curves.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to /content/training_curves.png")
else:
    print("No training history available")
```


## [26] markdown (106 chars)

```
## Section 9: Save and Download

Saves the model in 3 formats (LoRA, GGUF, merged) and downloads as a zip.
```


## [27] code (1463 chars)

```
# ============================================================
# 9.1 Save model in 3 formats with progress indicators
# ============================================================
os.makedirs(f"{OUTPUT_DIR}/lora", exist_ok=True)

# Save LoRA adapter
print("\n[1/3] Saving LoRA adapter...")
t0 = time.time()
model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")
lora_size = sum(p.stat().st_size for p in Path(f"{OUTPUT_DIR}/lora").rglob('*')) / 1e6
print(f"  LoRA saved ({lora_size:.1f}MB, {time.time()-t0:.1f}s)")

# Save GGUF
print("\n[2/3] Exporting GGUF (q4_k_m)...")
t0 = time.time()
model.save_pretrained_gguf(f"{OUTPUT_DIR}/gguf", tokenizer, quantization_method="q4_k_m")
gguf_size = sum(p.stat().st_size for p in Path(f"{OUTPUT_DIR}/gguf").rglob('*')) / 1e6
print(f"  GGUF saved ({gguf_size:.1f}MB, {time.time()-t0:.1f}s)")

# Save merged model
print("\n[3/3] Exporting merged model (16-bit)...")
t0 = time.time()
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged", tokenizer, save_method="merged_16bit")
merged_size = sum(p.stat().st_size for p in Path(f"{OUTPUT_DIR}/merged").rglob('*')) / 1e6
print(f"  Merged saved ({merged_size:.1f}MB, {time.time()-t0:.1f}s)")

print(f"\n{'='*60}")
print(f"  ALL FORMATS SAVED")
print(f"  LoRA:   {lora_size:.1f}MB")
print(f"  GGUF:   {gguf_size:.1f}MB")
print(f"  Merged: {merged_size:.1f}MB")
print(f"  Total:  {lora_size + gguf_size + merged_size:.1f}MB")
print(f"{'='*60}")
```


## [28] code (1067 chars)

```
# ============================================================
# 9.2 Zip and download with progress
# ============================================================
print("Creating zip archive...")
t0 = time.time()
shutil.make_archive(f"{WORK_DIR}/qwen4b-ctf-output", "zip", f"{WORK_DIR}/outputs")
zip_size = os.path.getsize(f"{WORK_DIR}/qwen4b-ctf-output.zip") / 1e6
print(f"  Created qwen4b-ctf-output.zip ({zip_size:.1f}MB, {time.time()-t0:.1f}s)")

if IS_COLAB:
    from google.colab import files
    print("\nStarting download to your machine...")
    files.download(f"{WORK_DIR}/qwen4b-ctf-output.zip")
    print("Download started! Check your browser downloads folder.")
elif IS_KAGGLE:
    print(f"Kaggle: outputs saved to {WORK_DIR}/qwen4b-ctf-output.zip \u2014 download from sidebar.")
else:
    print(f"Outputs saved to {WORK_DIR}/qwen4b-ctf-output.zip")
print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE")
print(f"  Model: {MODEL_NAME}")
print(f"  Output: {OUTPUT_DIR}/")
print(f"  Zip: {WORK_DIR}/qwen4b-ctf-output.zip ({zip_size:.1f}MB)")
print(f"{'='*60}")
```


## [29] markdown (48 chars)

```
## Section 10: (Optional) Test the Trained Model
```


## [30] code (776 chars)

```
# 10.1 Test inference - CTF prompt
from transformers import TextStreamer

messages = [
    {"role": "system", "content": SYSTEM_CTF},
    {"role": "user", "content": "Explain how a buffer overflow exploit works."},
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Qwen 3.5: use inner tokenizer (the returned object is a processor)
inference_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
inputs = inference_tokenizer(input_text, return_tensors="pt").to("cuda")
text_streamer = TextStreamer(inference_tokenizer, skip_prompt=True)
model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print("\nTest complete")
```


## [31] code (799 chars)

```
# 10.2 Test inference - coding prompt
messages = [
    {"role": "system", "content": SYSTEM_CODING},
    {"role": "user", "content": "Write a Python function to find the longest palindromic substring in O(n) using Manacher's algorithm."},
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Qwen 3.5: use inner tokenizer (the returned object is a processor)
inference_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
inputs = inference_tokenizer(input_text, return_tensors="pt").to("cuda")
text_streamer = TextStreamer(inference_tokenizer, skip_prompt=True)
model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
)
print("\nTest complete")
```

