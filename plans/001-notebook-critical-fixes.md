# Plan 001: Notebook Critical Fixes (N1-N6)

**Commit**: `5f04de4`  
**Status**: ✅ DONE (9b71fcc)  
**Effort**: S (~30 min)  
**Risk**: MED (notebook changes affect Colab users)

## Problem

The notebook has 6 bugs that prevent it from running correctly:
1. **N1 (CRITICAL)**: `hf_examples` and `doc_examples` are never defined — NameError at runtime
2. **N2**: `use_rslora` not passed to Gemma `get_peft_model` — quality-mode ignores rsLoRA
3. **N3**: Hardcoded `/content` paths break Kaggle support
4. **N4**: Generator propagates broken Cell 19 via pass-through
5. **N5**: (covered in Plan 003)
6. **N6**: `max_seq_length` not passed to Gemma `get_peft_model`

## Current State

**File**: `notebooks/qwen4b_self_contained.ipynb`

Cell 19 (Section 6.3) has:
```python
all_raw = all_writeups + hf_examples + doc_examples
```
But `hf_examples` and `doc_examples` are never defined. Cells 12 and 15 define `HF_DATASETS` and `DOC_SOURCES` as lists of tuples, but never loop through them to produce the actual examples.

Cell 21 (Section 7.3) Gemma path:
```python
if MODEL.startswith("gemma"):
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        # MISSING: use_rslora=USE_RSLORA
        # MISSING: max_seq_length=MAX_SEQ_LENGTH
        ...
    )
```

Cell 19 has hardcoded `/content` paths:
```python
sys.path.insert(0, "/content")  # should be WORK_DIR
```

## Fix

### Step 1: Add hf_examples and doc_examples generation to Cell 19

Before the `all_raw = ...` line, add:
```python
# Download HuggingFace datasets
hf_examples = []
for ds_name, max_samples in HF_DATASETS:
    try:
        ds = load_dataset(ds_name, split="train", streaming=True)
        for i, ex in enumerate(ds):
            if i >= max_samples:
                break
            qa = extract_qa(ex)
            if qa:
                hf_examples.append(qa)
    except Exception as e:
        print(f"  Failed to load {ds_name}: {e}")

# Scrape documentation
doc_examples = []
for doc_name, doc_url in DOC_SOURCES:
    try:
        doc_examples.extend(scrape_doc(doc_url))
    except Exception as e:
        print(f"  Failed to scrape {doc_name}: {e}")
```

### Step 2: Fix hardcoded /content paths in Cell 19

Replace:
```python
sys.path.insert(0, "/content")
```
With:
```python
sys.path.insert(0, WORK_DIR)
```

And fix the print:
```python
print(f"Saved to /content/data/merged/train.jsonl")
```
To:
```python
print(f"Saved to {WORK_DIR}/data/merged/train.jsonl")
```

### Step 3: Fix Cell 21 Gemma path — add use_rslora and max_seq_length

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=USE_RSLORA,        # ADD
    max_seq_length=MAX_SEQ_LENGTH, # ADD
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
```

### Step 4: Fix Cell 25 hardcoded print

Replace:
```python
print(f"Training curves saved to /content/training_curves.png")
```
With:
```python
print(f"Training curves saved to {WORK_DIR}/training_curves.png")
```

### Step 5: Update generator to match

In `scripts/generate_notebook.py`, the generator uses `code(H[19]["source"])` which will pick up the fixed cell automatically. No generator changes needed for N1/N3.

For N4, the generator's cell-19 builder (line 158) is a pass-through, so fixing the notebook fixes the generator output.

## Verification

```bash
# Verify notebook is valid JSON
python3 -c "import json; json.load(open('notebooks/qwen4b_self_contained.ipynb')); print('Valid JSON')"

# Verify hf_examples/doc_examples are defined before use
python3 -c "
import json
nb = json.load(open('notebooks/qwen4b_self_contained.ipynb'))
cell19 = ''.join(nb['cells'][19]['source'])
assert 'hf_examples = []' in cell19 or 'hf_examples =' in cell19, 'hf_examples not defined'
assert 'doc_examples = []' in cell19 or 'doc_examples =' in cell19, 'doc_examples not defined'
assert 'all_raw = all_writeups + hf_examples + doc_examples' in cell19
print('Cell 19: hf_examples and doc_examples defined')
"

# Verify no hardcoded /content in cell 19
python3 -c "
import json
nb = json.load(open('notebooks/qwen4b_self_contained.ipynb'))
cell19 = ''.join(nb['cells'][19]['source'])
assert '/content' not in cell19, 'Still has /content hardcoded'
print('Cell 19: no hardcoded /content')
"

# Verify Gemma path has use_rslora
python3 -c "
import json
nb = json.load(open('notebooks/qwen4b_self_contained.ipynb'))
cell21 = ''.join(nb['cells'][21]['source'])
assert 'use_rslora=USE_RSLORA' in cell21, 'use_rslora missing'
assert 'max_seq_length=MAX_SEQ_LENGTH' in cell21, 'max_seq_length missing'
print('Cell 21: use_rslora and max_seq_length present')
"
```

## Files to Modify

- `notebooks/qwen4b_self_contained.ipynb` (cells 19, 21, 25)
- `scripts/generate_notebook.py` (verify pass-through picks up fixes)
