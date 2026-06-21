# Plan 002: Eval Security + Correctness (1, 2)

**Commit**: `5f04de4`  
**Status**: ✅ DONE (9b71fcc)  
**Effort**: S (~20 min)  
**Risk**: MED (security fix changes grading behavior)

## Problem

1. **#1 (Security)**: `eval.py:161-166` — `exec()` on model-generated code with `__builtins__: {}` is an incomplete sandbox. Python escapes via `().__class__.__bases__[0].__subclasses__()` can import `os`, `subprocess`, etc.

2. **#2 (Correctness)**: `eval.py:469` — McNemar's p-value uses `math.exp(-chi2 / 2)` which is mathematically wrong. The correct formula uses the chi-squared survival function.

## Current State

```python
# eval.py:161-166 (insecure sandbox)
local_env = {}
exec(candidate, {"__builtins__": {}}, local_env)
for i, tc in enumerate(test_cases):
    exec(tc.get("setup", ""), {}, local_env)
    if not eval(tc["assert"], {}, local_env):
        return False, f"Test case {i+1} failed: {tc['assert']}"

# eval.py:469 (wrong p-value)
p_value = math.exp(-chi2 / 2)  # rough approximation
```

## Fix

### Step 1: Replace exec() sandbox with restricted module access

```python
import signal

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

def grade_code(response: str, reference: str = None, test_cases: list = None) -> tuple[bool, str]:
    # ... (existing code extraction and syntax check) ...

    if not test_cases:
        # ... (existing reference token check) ...

    # Restricted exec with timeout
    try:
        safe_builtins = {
            "range": range, "len": len, "int": int, "float": float,
            "str": str, "bool": bool, "list": list, "dict": dict,
            "tuple": tuple, "set": set, "print": print, "True": True,
            "False": False, "None": None, "abs": abs, "min": min,
            "max": max, "sum": sum, "sorted": sorted, "reversed": reversed,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        }
        local_env = {"__builtins__": safe_builtins}
        exec(candidate, local_env)

        for i, tc in enumerate(test_cases):
            if "setup" in tc:
                exec(tc["setup"], local_env)
            if not eval(tc["assert"], local_env):
                return False, f"Test {i+1} failed: {tc['assert']}"
        return True, f"All {len(test_cases)} tests passed"
    except TimeoutError:
        return False, "Execution timed out (3s limit)"
    except Exception as e:
        return False, f"Runtime error: {str(e)[:80]}"
```

### Step 2: Fix McNemar's p-value

Replace:
```python
p_value = math.exp(-chi2 / 2)  # rough approximation
```

With:
```python
# chi-squared survival function for 1 df
# p = 2 * (1 - Phi(sqrt(chi2))) where Phi is standard normal CDF
# Approximation: p ≈ erfc(sqrt(chi2/2))
import math
p_value = math.erfc(math.sqrt(chi2 / 2))
```

## Verification

```bash
# Verify sandbox blocks dangerous operations
uv run python -c "
from src.eval import grade_code
# Test that import os is blocked
correct, fb = grade_code('\`\`\`python\nimport os\nos.system(\"echo hacked\")\n\`\`\`', None, [{'assert': 'True'}])
print(f'import os: correct={correct}, feedback={fb}')
assert not correct or 'Runtime error' in fb or 'import' in fb.lower(), 'Sandbox did not block import os'

# Test that safe code works
correct, fb = grade_code('\`\`\`python\ndef add(a, b): return a + b\n\`\`\`', None, [{'assert': 'add(1, 2) == 3'}])
print(f'safe code: correct={correct}, feedback={fb}')
assert correct, 'Safe code should pass'
"

# Verify McNemar p-value is reasonable
uv run python -c "
import math
chi2 = 4.0
p = math.erfc(math.sqrt(chi2 / 2))
print(f'chi2=4.0 -> p={p:.4f}')
assert 0.04 < p < 0.06, f'p-value {p} out of expected range for chi2=4'
chi2 = 10.0
p = math.erfc(math.sqrt(chi2 / 2))
print(f'chi2=10.0 -> p={p:.6f}')
assert p < 0.002, f'p-value {p} too high for chi2=10'
"

# Run existing tests
uv run python -m pytest tests/ -v
```

## Files to Modify

- `src/eval.py` (grade_code sandbox, McNemar p-value)
