# Plan 022: Fix config drift between config.yaml and configs/*.yaml

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report.
>
> **Drift check (run first)**: `git diff --stat 087ce44..HEAD -- <in-scope paths>`
> If any in-scope file changed, compare excerpts against live code before proceeding.

## Status

- **Priority**: P0
- **Effort**: S
- **Risk**: MED
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `087ce44`, 2025-06-15

## Why this matters

`config.yaml` has `lora_alpha: 64` (alpha=2r) for gemma4, gemma4-12b, qwen35, ornith10. But `configs/*.yaml` has `lora_alpha: 32` (alpha=r). The CLI path uses configs/*.yaml, but any fallback through config.yaml gets different values. Unsloth recommends alpha==r for rsLoRA.

## Current state

- `config.yaml:24` → `lora_alpha: 64` for gemma4
- `configs/gemma4.yaml:16` → `lora_alpha: 32` for gemma4
- Same mismatch for gemma4-12b, qwen35, ornith10

## Scope

**In scope:** `config.yaml` only (5 model blocks to fix)

## Steps

### Step 1: Set alpha == r in config.yaml for all models

Edit `config.yaml` — for each of the 5 model blocks, change `lora_alpha: 64` to `lora_alpha: 32` (or match whatever `configs/<model>.yaml` has).

### Step 2: Verify consistency

```bash
for m in gemma4 gemma4-12b qwen35 qwen35-4b ornith10; do
  echo "=== $m ==="
  grep "lora_alpha" config.yaml configs/$m.yaml | head -2
done
```

**Verify:** `uv run python -m pytest tests/ -v --tb=short` → all pass

## Done criteria
- [ ] `config.yaml` lora_alpha values match `configs/*.yaml` for all 5 models
- [ ] All tests pass

## STOP conditions

- In-scope code changed since plan was written
- Verification fails twice after fix attempt
- Fix requires touching out-of-scope files

## Verification

```bash
uv run python -m pytest tests/ -v --tb=short
uv run ruff check src/ tests/
```
