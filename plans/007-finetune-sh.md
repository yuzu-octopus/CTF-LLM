# Plan 007: finetune.sh Fixes (10, 11)

**Commit**: `5f04de4`  
**Status**: ✅ DONE (9b71fcc)  
**Effort**: S (~15 min)  
**Risk**: LOW (shell script changes)

## Problem

1. **#10**: Step numbering out of order — `[1/5]`, `[2.5]`, `[2/5]`, `[3/5]`
2. **#11**: `--eval` not included in `--all` action — users must run eval separately

## Current State

```bash
# finetune.sh:22-36 (wrong numbering)
echo "[1/5] Building raw datasets..."
echo "[2.5] Building CTF-Dojo dataset..."
echo "[2/5] Generating synthetic rev/pwn examples..."
echo "[3/5] Processing to chat format..."

# finetune.sh:73 (--eval not in --all)
if [[ "$ACTION" == "--eval" ]]; then
    ...
fi
```

## Fix

### Step 1: Fix step numbering

```bash
echo "[1/6] Building raw datasets..."
echo "[2/6] Building CTF-Dojo dataset..."
echo "[3/6] Downloading HuggingFace datasets..."
echo "[4/6] Generating synthetic rev/pwn examples..."
echo "[5/6] Processing to chat format..."
echo "[6/6] Merging datasets..."
```

### Step 2: Add --eval to --all

```bash
# After the train block, add:
if [[ "$ACTION" == "--eval" ]] || [[ "$ACTION" == "--all" ]]; then
    echo ""
    echo "[7/7] Running CTF evaluation..."
    ADAPTER="outputs/${MODEL}-ctf/lora"
    if [[ -d "$ADAPTER" ]]; then
        uv run src/eval.py --model "$MODEL" --adapter "$ADAPTER"
    else
        echo "  No adapter found at $ADAPTER — skipping eval"
    fi
fi
```

## Verification

```bash
# Verify step numbering is sequential
grep -o '\[[0-9]*/[0-9]*\]' finetune.sh | sort -t'/' -k1 -n
# Expected: [1/6] [2/6] [3/6] [4/6] [5/6] [6/6] [7/7]

# Verify --all includes eval
grep -A2 "ACTION.*--all" finetune.sh | grep eval
# Expected: eval block triggers on --all
```

## Files to Modify

- `finetune.sh`
