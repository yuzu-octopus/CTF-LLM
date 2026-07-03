# Plan 025: Fix temperature=None in model.generate()

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report.
>
> **Drift check (run first)**: `git diff --stat 087ce44..HEAD -- <in-scope paths>`
> If any in-scope file changed, compare excerpts against live code before proceeding.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `087ce44`, 2025-06-15

## Why this matters

`src/eval.py:104` passes `temperature=None` when `n_samples=1`. Some quantized model backends reject non-numeric temperature values.

## Current state

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=(n_samples > 1),
    temperature=0.6 if n_samples > 1 else None,
)
```

## Scope

**In scope:** `src/eval.py` only

## Steps

### Step 1: Conditionally build generate kwargs

```python
gen_kwargs = {
    "max_new_tokens": max_new_tokens,
    "do_sample": (n_samples > 1),
}
if n_samples > 1:
    gen_kwargs["temperature"] = 0.6

outputs = model.generate(**inputs, **gen_kwargs)
```

**Verify:** `uv run python -m pytest tests/test_eval.py -v` → all pass

## Done criteria
- [ ] No `temperature=None` in eval.py
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
